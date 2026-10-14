class StageFailure(RuntimeError):
    def __init__(self, stage: str, detail: object, hdu_index: int | None = None):
        self.stage = stage
        self.detail = str(detail)
        self.hdu_index = hdu_index
        super().__init__(stage, self.detail)

    def __str__(self) -> str:
        prefix = f"HDU {self.hdu_index}: " if self.hdu_index is not None else ""
        return f"{prefix}{self.stage} failed: {self.detail}"
