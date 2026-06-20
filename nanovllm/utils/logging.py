from time import perf_counter


class Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.run_start = None
        self.start_times = {}
        self.ttft_list = []
        self.latency_list = []
        self.total_output_tokens = 0
        # summary
        self.total_requests = 0
        self.finished_requests = 0
        self.summary_written = False
        self.ttft_by_seq = {}
        # prefix kv cache
        self.total_prefill_computed_tokens = 0
        self.total_prefill_cached_tokens = 0

    def add_request(self, seq):
        if self.run_start is None:
            self.run_start = perf_counter()
        self.total_requests += 1
        self.start_times[seq.seq_id] = perf_counter()
        self.write(
            f"request: seq={seq.seq_id}, "
            f"prompt_tokens={len(seq.prompt_token_ids)}"
        )

    def before_step(self, waiting, running, seqs, is_prefill):
        if not is_prefill:
            return

        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        cached_tokens = sum(seq.num_cached_tokens for seq in seqs)
        total_tokens = cached_tokens + num_tokens
        hit_rate = cached_tokens / total_tokens if total_tokens else 0.0
        seq_ids = [seq.seq_id for seq in seqs]

        self.write(
            f"prefill: batch={len(seqs)}, "
            f"seqs={seq_ids}, "
            f"computed_tokens={num_tokens}, "
            f"cached_tokens={cached_tokens}, "
            f"prefix_hit_rate={hit_rate:.2%}"
        )

    def after_step(self, seqs):
        now = perf_counter()

        for seq in seqs:
            if seq.num_completion_tokens == 1:
                ttft = now - self.start_times[seq.seq_id]
                self.ttft_by_seq[seq.seq_id] = ttft
                self.ttft_list.append(ttft)
                self.write(f"ttft: seq={seq.seq_id}, {ttft:.4f}s")

            if seq.is_finished:
                latency = now - self.start_times[seq.seq_id]
                self.latency_list.append(latency)
                self.total_output_tokens += seq.num_completion_tokens
                self.finished_requests += 1

                ttft = self.ttft_by_seq.get(seq.seq_id, 0.0)
                self.write(
                    f"finish: seq={seq.seq_id}, "
                    f"output_tokens={seq.num_completion_tokens}, "
                    f"ttft={ttft:.4f}s, "
                    f"latency={latency:.4f}s"
                )

        if (
            self.total_requests > 0
            and self.finished_requests == self.total_requests
            and not self.summary_written
        ):
            self.summary_written = True
            self.write_summary(now)

    def write_summary(self, now):
        elapsed = max(now - self.run_start, 1e-9)
        tps = self.total_output_tokens / elapsed

        prefix_total = self.total_prefill_cached_tokens + self.total_prefill_computed_tokens
        prefix_hit_rate = self.total_prefill_cached_tokens / prefix_total if prefix_total else 0.0

        self.write(
            f"summary: requests={self.finished_requests}, "
            f"output_tokens={self.total_output_tokens}, "
            f"elapsed={elapsed:.4f}s, "
            f"tps={tps:.2f}tok/s, "
            f"ttft_p95={self._percentile(self.ttft_list, 0.95):.4f}s, "
            f"ttft_p99={self._percentile(self.ttft_list, 0.99):.4f}s, "
            f"latency_p95={self._percentile(self.latency_list, 0.95):.4f}s, "
            f"latency_p99={self._percentile(self.latency_list, 0.99):.4f}s,"
            f"prefix_cached_tokens={self.total_prefill_cached_tokens}, "
            f"prefix_computed_tokens={self.total_prefill_computed_tokens}, "
            f"prefix_hit_rate={prefix_hit_rate:.2%}"
        )

    def _percentile(self, values, q):
        if not values:
            return 0.0
        values = sorted(values)
        pos = (len(values) - 1) * q
        low = int(pos)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (pos - low)

    def write(self, text):
        print(text)
        with open(self.log_file, "a") as f:
            f.write(text + "\n")
