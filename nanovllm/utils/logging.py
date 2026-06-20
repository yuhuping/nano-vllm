from time import perf_counter


class Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.cur_step = None        # 当前step 是 prefill 还是 decode
        self.cur_num_seqs = 0       # 当前 step 处理的 seq 数
        self.cur_num_tokens = 0     # 当前 step 处理的 token 数
        self.cur_seq_waiting = []    # 当前 step 处理的 waiting 状态的seq（ 以前 5 个字符显示）
        self.cur_seq_running = []
        self.ttft_list = []         # 每条 seq 的 ttft，单位是秒
        self.start_times = {}

    def add_request(self, seq):
        self.start_times[seq.seq_id] = perf_counter()

    def before_step(self, waiting, running, seqs, is_prefill):
        self.cur_step = "prefill" if is_prefill else "decode"
        self.cur_num_seqs = len(seqs)
        self.cur_num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        self.cur_seq_waiting = [self._seq(seq) for seq in waiting]
        self.cur_seq_running = [self._seq(seq) for seq in running]
        self.write(f"{self.cur_step}: seqs={self.cur_num_seqs}, tokens={self.cur_num_tokens}, waiting={self.cur_seq_waiting}, running={self.cur_seq_running}")

    def after_step(self, seqs):
        for seq in seqs:
            if seq.num_completion_tokens == 1:
                ttft = perf_counter() - self.start_times[seq.seq_id]
                self.ttft_list.append(ttft)
                self.write(f"ttft: seq={seq.seq_id}, {ttft:.4f}s")

    def _seq(self, seq):
        return f"{seq.seq_id}:{seq.token_ids[:5]}"

    def write(self, text):
        print(text)
        with open(self.log_file, "a") as f:
            f.write(text + "\n")
