from collections import defaultdict
from datetime import datetime

class Blockchain:
    def __init__(self):
        self.chain = []
        self.suspicion_log = defaultdict(int)

    def add_block(self, cid, round_num, distance):
        block = {
            "timestamp": datetime.utcnow().isoformat(),
            "cid": cid,
            "round": round_num,
            "distance": distance
        }
        self.chain.append(block)
        self.suspicion_log[cid] += distance  # distance 누적

    def should_penalize(self, cid, threshold=3.0):
        """누적 distance 가 threshold 이상이면 패널티"""
        return self.suspicion_log[cid] >= threshold
