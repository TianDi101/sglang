"""E2E contract for an external-cache linker whose remote KV reads fail.

Pairs with ``srt/mem_cache/unified_cache/linker_fault_injection.py``. A model
test mixes this in on top of the linker KL test it already has, so the failure
arm reuses that arm's launch configuration verbatim -- same pools, same page
size, same eviction pressure, so the same load-backs happen.

What is being pinned
--------------------
A failed remote KV read must not reach the client as an answer. It can cost a
request an attempt, and the engine has to survive it, but the one outcome the
fix exists to prevent is a request that loads KV which never arrived and then
generates from it at HTTP 200.

Accuracy is how that is measured, and gsm8k is what makes it measurable: every
question has a known-correct answer, so a request served over KV that never
arrived shows up as a *wrong* answer. That is why this is a stronger check than
comparing token ids between two runs -- runs taking different cache paths
diverge on random token ids with nothing wrong, so a diff cannot separate
corruption from a different-but-valid path. Ground truth can.

The eval is ``run_eval(eval_name="gsm8k")``, unmodified. Its ``GenerateSampler``
retries a failed request six times with exponential backoff and returns an empty
string only if all six are exhausted, so an aborted request is retried rather
than lost. At the injection rates used here a question needs six consecutive
failed load batches to score zero, which is why this arm is held to the *same*
accuracy threshold as the healthy one rather than a relaxed one: the retry
absorbs the aborts, and anything left is corruption.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

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
    # the sampler's retry still absorbs it.
    linker_load_failure_prob: float = 0.05

    # gsm8k settings. num_shots matches the healthy arm; GSM8KEval's own
    # default is 5.
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
        Unlike the eval, this probe does not retry, so it sees the abort the
        eval's sampler would have hidden.

        Finding no abort at all is a failure and not a pass: it means the
        injector never fired, or the requests never reached the linker, and
        either way the accuracy assertion would have been measuring an
        uninjected run. This is the sensitivity control, and it is the thing a
        fault test most often silently loses.
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

    def test_gsm8k_accuracy_under_linker_load_failure(self):
        """A failed load may cost an attempt. It must never change an answer."""
        from sglang.test.run_eval import run_eval

        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                eval_name="gsm8k",
                # SGLang-native /generate, the transport the healthy arm uses.
                api="generate",
                num_examples=self.num_gsm8k_questions,
                num_threads=self.gsm8k_parallel,
                num_shots=self.gsm8k_num_shots,
                max_tokens=self.gsm8k_max_new_tokens,
                # Required, and only by the report writer at the very end:
                # it names the dump file `sampler.model.replace("/", "_")`,
                # so leaving it None raises AttributeError *after* the score
                # has been computed and printed -- losing a finished eval to
                # a filename. GenerateSampler does not resolve it from the
                # server the way the OpenAI-shaped samplers can.
                model=getattr(self, "model", None) or "linker-load-failure",
            )
        )
        print(
            f"[{type(self).__name__}] gsm8k at "
            f"prob={self.linker_load_failure_prob}: score={metrics['score']:.3f} "
            f"(threshold {self.gsm8k_threshold})"
        )

        self.assertGreaterEqual(
            metrics["score"],
            self.gsm8k_threshold,
            "gsm8k accuracy fell under injected linker load failures. The "
            "sampler retries a failed request six times, so aborts alone "
            "should not move this number -- a drop means questions were "
            "answered from KV that never arrived.",
        )

        response = requests.get(self.base_url + "/health_generate", timeout=120)
        self.assertEqual(response.status_code, 200, "engine died during the eval")
