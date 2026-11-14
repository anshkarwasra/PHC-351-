"""Structured results so an agent can parse output instead of reading prints."""
import json
from dataclasses import dataclass, asdict, field


@dataclass
class CheckResult:
    step: int
    name: str
    status: str            # "pass" | "fail" | "warn"
    measured: object
    threshold: object
    detail: str = ""


@dataclass
class Report:
    results: list = field(default_factory=list)

    def add(self, r: CheckResult):
        self.results.append(r)

    def summary(self):
        counts = {"pass": 0, "fail": 0, "warn": 0}
        for r in self.results:
            counts[r.status] += 1
        return counts

    def write(self, path):
        with open(path, "w") as fh:
            json.dump([asdict(r) for r in self.results], fh, indent=1, default=str)

    def exit_code(self):
        return 1 if self.summary()["fail"] else 0