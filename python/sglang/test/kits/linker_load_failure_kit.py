"""E2E contract for an external-cache linker whose remote KV reads fail.

Pairs with ``srt/mem_cache/unified_cache/linker_fault_injection.py``. A model
test mixes this in on top of the linker KL test it already has, so the failure
arm reuses that arm's launch configuration verbatim -- same pools, same page
size, same eviction pressure, so the same load-backs happen.

What is being pinned
--------------------
A failed remote KV read must not reach the client as an answer. It can abort
the request, and the engine has to survive it, but the one outcome the fix
exists to prevent is a request that loads KV which never arrived and then
generates from it at HTTP 200.

Accuracy is how that is measured, and gsm8k is what makes it measurable:
every question has a known-correct answer, so a request served over KV that
never arrived shows up as a *wrong* answer rather than a missing one. That
distinction is the whole assertion --

    correct + aborted >= threshold

-- and it is why this is a stronger check than comparing token ids between two
runs. Two runs that take different cache paths diverge on random token ids
even with nothing wrong, so a token diff cannot separate corruption from a
different-but-valid path. Ground truth can.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import requests

# The message _mark_failed_linker_loads puts on the abort.
EXTERNAL_KV_LOAD_ABORT_TEXT = "external KV cache load failed"


class LinkerLoadFailureMixin:
    """gsm8k accuracy and the abort contract, under injected load failures.

    The mixin owns no server. Combine it with a class that launches one with
    ``SGLANG_TEST_LINKER_LOAD_FAILURE_PROB`` set, entered *before* the launch so
    the server subprocesses inherit it.
    """

    # Matches the disaggregation failure test's rate. High enough that the
    # abort path is hit many times over a 200-question eval, low enough that
    # most requests still complete and accuracy stays measurable.
    linker_load_failure_prob: float = 0.05

    # Of the questions asked, this fraction must come back either correct or
    # aborted. The complement is the budget for answers that were served and
    # wrong -- which is what serving a failed load looks like.
    correct_or_aborted_threshold: float = 0.93
    num_gsm8k_questions: int = 200
    gsm8k_num_shots: int = 10
    gsm8k_max_new_tokens: int = 512
    gsm8k_parallel: int = 32

    # Bound for the abort probe below. At the default 5% this leaves a ~0.05%
    # chance of finding no abort in a healthy run, and it usually stops after
    # ~20 requests.
    max_abort_probes: int = 150
    abort_probe_prefix_len: int = 2048

    def _generate_one(self, input_ids, max_new_tokens=1):
        """Send one request and classify it, tolerating an abort either way."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=600,
        )
        # An abort is a 500 in some versions and a 200 whose finish_reason says
        # "abort" in others. Classifying on HTTP status alone reports a correct
        # abort as a served request.
        if response.status_code != 200:
            return "aborted", response.text
        body = response.json()
        meta = body.get("meta_info") or {}
        finish_reason = meta.get("finish_reason") or {}
        if finish_reason.get("type") == "abort":
            return "aborted", str(finish_reason)
        return "served", body

    def _flush(self):
        # /flush_cache legitimately 400s while requests are in flight.
        for _ in range(10):
            response = requests.post(
                self.base_url + "/flush_cache", params={"timeout": 30}, timeout=60
            )
            if response.status_code == 200 and "not flushed" not in response.text:
                return
        raise AssertionError(f"could not flush the cache: {response.text}")

    def test_load_failures_abort_and_the_engine_survives(self):
        """The abort contract, and the control that says injection was live.

        Sends requests that have to come back through the linker -- warm the
        store, drop the device tree, ask again -- until one of them is aborted.

        Finding no abort at all is a failure and not a pass: it means the
        injector never fired, or the requests never reached the linker, and
        either way every other assertion here would have been measuring
        nothing. This is the sensitivity control, and it is the thing a fault
        test most often silently loses.
        """
        rng = random.Random(20260904)
        served_after_abort = 0
        aborted = 0
        probes = 0

        while probes < self.max_abort_probes and (
            aborted == 0 or served_after_abort == 0
        ):
            prompt = [rng.randint(1, 30000) for _ in range(self.abort_probe_prefix_len)]
            # Warm: this populates the external store.
            outcome, _ = self._generate_one(prompt)
            probes += 1
            # Drop the device tree so the repeat has to come from the store.
            self._flush()
            outcome, detail = self._generate_one(prompt)
            probes += 1

            if outcome == "aborted":
                aborted += 1
                self.assertIn(
                    EXTERNAL_KV_LOAD_ABORT_TEXT,
                    str(detail),
                    "a request aborted for some reason other than the linker "
                    f"load failure under test: {detail}",
                )
            elif aborted:
                # Whatever the abort tore down, the next request still works.
                served_after_abort += 1

        print(
            f"[{type(self).__name__}] abort probe: {aborted} aborted, "
            f"{served_after_abort} served after an abort, {probes} requests"
        )
        self.assertGreater(
            aborted,
            0,
            f"No request was aborted in {probes} requests at "
            f"prob={self.linker_load_failure_prob}. The injection did not "
            "reach the linker, so this run proves nothing -- do not read a "
            "pass here as evidence the failure path works.",
        )
        self.assertGreater(
            served_after_abort,
            0,
            "the engine did not serve a request after an aborted one",
        )
        response = requests.get(self.base_url + "/health_generate", timeout=120)
        self.assertEqual(response.status_code, 200, "engine died on a load failure")

    def _gsm8k_questions(self):
        """Few-shot prompts and their known answers.

        Reuses the prompt construction and answer parsing from
        ``few_shot_gsm8k``; only that module's ``run_eval`` driver is
        deprecated, and this does not use it -- see
        ``test_gsm8k_accuracy_under_linker_load_failure`` for why it cannot.
        """
        from sglang.test.few_shot_gsm8k import (
            get_answer_value,
            get_few_shot_examples,
            get_one_example,
        )
        from sglang.utils import download_and_cache_file, read_jsonl

        lines = list(
            read_jsonl(
                download_and_cache_file(
                    "https://raw.githubusercontent.com/openai/grade-school-math"
                    "/master/grade_school_math/data/test.jsonl"
                )
            )
        )
        few_shot = get_few_shot_examples(lines, self.gsm8k_num_shots)
        return [
            (
                few_shot + get_one_example(lines, i, False),
                get_answer_value(lines[i]["answer"]),
            )
            for i in range(min(self.num_gsm8k_questions, len(lines)))
        ], get_answer_value

    def _ask_one_question(self, prompt):
        """One gsm8k question. Returns its text, or None if it was aborted."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": self.gsm8k_max_new_tokens,
                    "stop": ["Question", "Assistant:", "<|separator|>"],
                },
            },
            timeout=1200,
        )
        if response.status_code != 200:
            return None
        body = response.json()
        finish_reason = (body.get("meta_info") or {}).get("finish_reason") or {}
        if finish_reason.get("type") == "abort":
            return None
        return body.get("text", "")

    def test_gsm8k_accuracy_under_linker_load_failure(self):
        """A failed load may cost an answer. It must never change one.

        Asks the questions directly rather than through
        ``few_shot_gsm8k.run_eval`` or ``run_eval``, because both give up the
        moment a request fails, and requests failing is the entire point here.
        The sgl-lang path in particular stores the error on the state and never
        fills the variable, so reading ``states[i]["answer"]`` raises KeyError
        on the first abort and the whole eval is lost -- the accuracy number
        that matters would never be computed. Neither harness retries, so a
        request aborted once stays aborted.

        Asking directly also separates *aborted* from *unparseable*, which the
        harnesses fold together into ``invalid``. That distinction is load
        bearing: garbage generated over KV that never arrived can easily fail
        to parse, and counting it as "invalid" would forgive exactly the
        outcome under test.
        """
        questions, get_answer_value = self._gsm8k_questions()
        aborted = correct = wrong = 0

        with ThreadPoolExecutor(max_workers=self.gsm8k_parallel) as pool:
            answers = pool.map(
                self._ask_one_question, [prompt for prompt, _ in questions]
            )
            for (_, label), answer in zip(questions, answers):
                if answer is None:
                    aborted += 1
                elif get_answer_value(answer) == label:
                    correct += 1
                else:
                    wrong += 1

        total = len(questions)
        print(
            f"[{type(self).__name__}] gsm8k at "
            f"prob={self.linker_load_failure_prob}: correct={correct}/{total} "
            f"aborted={aborted} wrong={wrong}"
        )

        self.assertGreaterEqual(
            (correct + aborted) / total,
            self.correct_or_aborted_threshold,
            f"{wrong}/{total} questions were answered, and answered wrongly, "
            "under injected load failures. A failed load is allowed to cost an "
            "answer; it is not allowed to change one. Serving KV that never "
            "arrived is what this looks like.",
        )

        response = requests.get(self.base_url + "/health_generate", timeout=120)
        self.assertEqual(response.status_code, 200, "engine died during the eval")
