from typing import Optional
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    PLM: str = field(
        default="sh110495/kor-pegasus",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    save_path: str = field(
        default="checkpoints", 
        metadata={
            "help": "Path to save checkpoint from fine tune model"
        },
    )
 
@dataclass
class DataArguments:
    max_input_length: Optional[int] = field(
        default=1024, 
        metadata={
            "help": "Max length of input sequence"
        },
    )
    max_target_length: Optional[int] = field(
        default=256, 
        metadata={
            "help": "Max length of target sequence"
        },
    )
    dir_path: str = field(
        default="dataset", 
        metadata={
            "help": "Path to dataset"
        }
    )

@dataclass
class TrainingArguments:
    seed: Optional[int] = field(
        default=42,
        metadata={
            "help": "Random seed for initialization"
        },
    )
    batch_size: Optional[int] = field(
        default=256, 
        metadata={
            "help": "Training batch size"
        }
    )
    epochs: Optional[int] = field(
        default=10,
        metadata={
            "help": "Number of training epochs"
        }
    )
    learning_rate: Optional[float] = field(
        default=1e-4,
        metadata={
            "help": "Learning rate"
        }
    )
    warmup_ratio: Optional[float] = field(
        default=1e-2,
        metadata={
            "help": "Warmup ratio"
        }
    )
    weight_decay: Optional[float] = field(
        default=1e-2,
        metadata={
            "help": "Weight decay"
        }
    )
    max_steps: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Max training steps"
        }
    )
    save_steps: Optional[int] = field(
        default=10000,
        metadata={
            "help": "Save steps during training"
        }
    )

@dataclass
class LoggingArguments:
    dotenv_path: Optional[str] = field(
        default="wandb.env", 
        metadata={
            "help": "input your dotenv path"
        },
    )
    project_name: Optional[str] = field(
        default="pegasus",
         metadata={
            "help": "project name"
        },
    )
    group_name: Optional[str] = field(
        default="base-model", 
        metadata={
            "help": "group name"
        },
    )

