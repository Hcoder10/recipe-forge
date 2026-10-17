"""Token accounting tests; no GPU or vLLM process is needed."""
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from forge.tools import benchjob


def stream(usage=None, chat=False):
    events = [{"choices": [{"delta": {"role": "assistant"}}]}]
    for piece in ("hello", " world"):
        choice = {"delta": {"content": piece}} if chat else {"text": piece}
        events.append({"choices": [choice]})
    if usage is not None:
        events.append({"choices": [], "usage": usage})
    data = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return io.BytesIO((data + "data: [DONE]\n\n").encode())


def bad_request(code=400):
    return urllib.error.HTTPError("http://localhost", code, "Rejected", {}, io.BytesIO())


class BenchjobTokensTests(unittest.TestCase):
    def test_fixed_length_body_for_both_endpoints(self):
        for chat in (None, {"on": True}):
            with self.subTest(chat=chat), patch.object(
                benchjob.urllib.request, "urlopen", return_value=stream({"completion_tokens": 80}, bool(chat))
            ) as opened:
                result = benchjob.stream_one(8000, "model", {"prompt": "hi", "max_tokens": 80}, 100, chat)
                request = opened.call_args.args[0]
                body = json.loads(request.data)
                self.assertEqual(body["max_tokens"], 80)
                self.assertEqual(body["min_tokens"], 80)
                self.assertIs(body["ignore_eos"], True)
                self.assertEqual(body["temperature"], 0)
                self.assertTrue(body["stream_options"]["include_usage"])
                self.assertTrue(request.full_url.endswith("/chat/completions" if chat else "/v1/completions"))
                self.assertEqual(result["tokens"], 80)
                self.assertEqual(result["streamed_content_pieces"], 2)
                self.assertTrue(result["fixed_length"])
                self.assertFalse(result["token_count_mismatch"])

    def test_mismatch_threshold_and_zero_usage(self):
        for count, mismatch in ((100, False), (99, False), (101, False), (98, True), (102, True), (0, True)):
            with self.subTest(count=count), patch.object(
                benchjob.urllib.request, "urlopen", return_value=stream({"completion_tokens": count})
            ):
                result = benchjob.stream_one(8000, "model", {}, 100, None)
                self.assertEqual(result["tokens"], count)
                self.assertEqual(result["completion_tokens"], count)
                self.assertEqual(result["requested_tokens"], 100)
                self.assertEqual(result["token_count_mismatch"], mismatch)

    def test_missing_usage_uses_requested_length(self):
        for usage in (None, {}, {"prompt_tokens": 5}, {"completion_tokens": None}):
            with self.subTest(usage=usage), patch.object(
                benchjob.urllib.request, "urlopen", return_value=stream(usage)
            ):
                result = benchjob.stream_one(8000, "model", {}, 100, None)
                self.assertEqual(result["tokens"], 100)
                self.assertEqual(result["streamed_content_pieces"], 2)
                self.assertIsNone(result["completion_tokens"])
                self.assertFalse(result["token_count_mismatch"])

    def test_400_retries_without_sampling_extras(self):
        for chat in (None, {"on": True}):
            for usage, expected in ((None, 2), ({"completion_tokens": 7}, 7)):
                with self.subTest(chat=chat, usage=usage), patch.object(
                    benchjob.urllib.request, "urlopen", side_effect=[bad_request(), stream(usage, bool(chat))]
                ) as opened:
                    result = benchjob.stream_one(8000, "model", {"prompt": "hi"}, 100, chat)
                    self.assertEqual(opened.call_count, 2)
                    first, retry = [json.loads(call.args[0].data) for call in opened.call_args_list]
                    first.pop("min_tokens")
                    first.pop("ignore_eos")
                    self.assertEqual(first, retry)
                    self.assertFalse(result["fixed_length"])
                    self.assertFalse(result["token_count_mismatch"])
                    self.assertEqual(result["tokens"], expected)

    def test_retry_is_bounded_and_other_errors_propagate(self):
        for errors in ([bad_request(), bad_request()], [bad_request(500)]):
            with self.subTest(attempts=len(errors)), patch.object(
                benchjob.urllib.request, "urlopen", side_effect=errors
            ) as opened:
                with self.assertRaises(urllib.error.HTTPError):
                    benchjob.stream_one(8000, "model", {}, 100, None)
                self.assertEqual(opened.call_count, len(errors))

    def test_workload_exposes_per_request_counts_and_flags(self):
        with patch.object(benchjob.urllib.request, "urlopen", side_effect=[
            stream({"completion_tokens": 16}),  # warmup
            stream({"completion_tokens": 90}),
            bad_request(), stream(),
        ]):
            result = benchjob.run_workload(8000, "model", [{"prompt": "a"}, {"prompt": "b"}], {"max_tokens": 100})
        self.assertEqual(result["tokens"], 92)
        self.assertEqual([r["tokens"] for r in result["per_request"]], [90, 2])
        self.assertEqual([r["token_count_mismatch"] for r in result["per_request"]], [True, False])
        self.assertTrue(result["token_count_mismatch"])
        self.assertFalse(result["fixed_length"])
        self.assertIn("per_prompt_decode", result)
        self.assertIn("decode_tok_s", result)
        self.assertIn("output_tok_s", result)


if __name__ == "__main__":
    unittest.main()
