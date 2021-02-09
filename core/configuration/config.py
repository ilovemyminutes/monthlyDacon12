from dataclasses import dataclass # 데이터만을 담는 객체 생성을 편리하게 하기 위한 클래스

@dataclass
class Config:
    device: str
    epochs: int
    batch_size: int
    lr: float