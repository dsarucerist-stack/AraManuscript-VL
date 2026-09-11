import gc
import os

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

# Hugging Face model ID or local checkpoint path.
#
# Default:
# Qwen/Qwen2.5-VL-3B-Instruct
#
# Example:
# MODEL_PATH=/path/to/checkpoint
#
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)

# Hugging Face token.
#
# Required only if the model/checkpoint is private.
#
# Example:
# export HF_TOKEN="hf_xxxxxxxxxxxxx"
#
HF_TOKEN = os.getenv("HF_TOKEN")


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

# Training CSV.
#
# Expected columns:
#   file_name
#   text
#   type (optional)
#
CSV_TRAIN = os.getenv(
    "CSV_TRAIN",
    "./data/train.csv",
)

# Validation CSV.
CSV_VAL = os.getenv(
    "CSV_VAL",
    "./data/validation.csv",
)

# Directory containing the manuscript images.
IMAGE_DIR = os.getenv(
    "IMAGE_DIR",
    "./data/images",
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

# Directory used to save checkpoints and the final model.
OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    "./outputs/qwen2.5-vl-3b-arabic-manuscript-lora",
)


# ------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------

BATCH_SIZE = int(
    os.getenv("BATCH_SIZE", "1")
)

GRAD_ACCUM = int(
    os.getenv("GRAD_ACCUM", "16")
)

EPOCHS = int(
    os.getenv("EPOCHS", "2")
)

LR = float(
    os.getenv("LR", "1e-5")
)


# ------------------------------------------------------------
# Hardware
# ------------------------------------------------------------

# Optional CUDA device selection.
#
# Example:
#
# CUDA_VISIBLE_DEVICES=0
#
# or:
#
# CUDA_VISIBLE_DEVICES=0,1
#
CUDA_VISIBLE_DEVICES = os.getenv(
    "CUDA_VISIBLE_DEVICES"
)

if CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES


# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------

PROMPT = os.getenv(
    "PROMPT"
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print()
print("=" * 70)
print("Qwen2.5-VL Arabic Manuscript Recognition")
print("LoRA Fine-Tuning")
print("=" * 70)

print(f"Model:              {MODEL_PATH}")
print(f"Training CSV:       {CSV_TRAIN}")
print(f"Validation CSV:     {CSV_VAL}")
print(f"Image directory:    {IMAGE_DIR}")
print(f"Output directory:   {OUTPUT_DIR}")
print(f"Batch size:         {BATCH_SIZE}")
print(f"Gradient accum.:    {GRAD_ACCUM}")
print(f"Epochs:             {EPOCHS}")
print(f"Learning rate:      {LR}")

if CUDA_VISIBLE_DEVICES:
    print(f"CUDA devices:       {CUDA_VISIBLE_DEVICES}")
else:
    print("CUDA devices:       default")

print("=" * 70)
print()


# ============================================================
# MEMORY MANAGEMENT
# ============================================================

def clear_memory():
    """
    Release unused CPU and GPU memory.
    """

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("🚀 Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    token=HF_TOKEN,
    trust_remote_code=True,
)

# Use left padding for generation/training.
processor.tokenizer.padding_side = "left"

clear_memory()


# ============================================================
# LOAD MODEL
# ============================================================

print("🚀 Loading model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("Base model loaded.")


# ============================================================
# LoRA CONFIGURATION
# ============================================================

print("🔧 Configuring LoRA...")

lora_config = LoraConfig(
    r=128,
    lora_alpha=256,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    lora_dropout=0.05,

    bias="none",

    task_type=TaskType.CAUSAL_LM,
)


# Apply LoRA
model = get_peft_model(
    model,
    lora_config,
)


# Display trainable parameters
model.print_trainable_parameters()

model.train()

print("✅ LoRA configuration applied.")


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(file_name):
    """
    Find an image using common image extensions.

    The CSV can contain:
        image.png

    or:
        image

    Supported extensions:
        .jpg
        .jpeg
        .png
        .JPG
        .JPEG
        .PNG
    """

    file_name = str(file_name)

    # --------------------------------------------------------
    # First try the filename exactly as provided.
    # --------------------------------------------------------

    direct_path = os.path.join(
        IMAGE_DIR,
        file_name,
    )

    if os.path.exists(direct_path):
        return direct_path

    # --------------------------------------------------------
    # If no extension is provided, try common extensions.
    # --------------------------------------------------------

    base_name = os.path.splitext(file_name)[0]

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG",
    ]

    for ext in extensions:

        path = os.path.join(
            IMAGE_DIR,
            base_name + ext,
        )

        if os.path.exists(path):
            return path

    return None


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image_path):
    """
    Load an image and convert it to RGB.
    """

    return Image.open(
        image_path
    ).convert("RGB")


# ============================================================
# SAMPLE PREPROCESSING
# ============================================================

def preprocess_sample(image_path, text):
    """
    Convert one manuscript image + transcription pair
    into Qwen2.5-VL training inputs.

    The user prompt is masked from the loss.
    The loss is therefore calculated only on the
    assistant's Arabic transcription.
    """

    try:

        image = prepare_image(
            image_path
        )

        # ----------------------------------------------------
        # Full conversation
        # ----------------------------------------------------

        messages_full = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    }
                ],
            },
        ]

        # ----------------------------------------------------
        # Question only
        #
        # Used to determine which tokens belong to the
        # input prompt and should therefore be ignored
        # during loss calculation.
        # ----------------------------------------------------

        messages_question = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ]

        # ----------------------------------------------------
        # Apply Qwen chat templates
        # ----------------------------------------------------

        full_text = processor.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
        )

        question_text = processor.apply_chat_template(
            messages_question,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ----------------------------------------------------
        # Process image
        # ----------------------------------------------------

        image_inputs, _ = process_vision_info(
            messages_full
        )

        # ----------------------------------------------------
        # Process complete conversation
        # ----------------------------------------------------

        inputs = processor(
            text=[full_text],
            images=image_inputs,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        )

        # ----------------------------------------------------
        # Process user question only
        # ----------------------------------------------------

        question_inputs = processor(
            text=[question_text],
            images=image_inputs,
            return_tensors="pt",
        )

        question_len = (
            question_inputs.input_ids.shape[1]
        )

        # ----------------------------------------------------
        # Create labels
        # ----------------------------------------------------

        labels = inputs.input_ids.clone()

        # Ignore the user prompt.
        #
        # Only the assistant's transcription contributes
        # to the training loss.

        labels[:, :question_len] = -100

        # ----------------------------------------------------
        # Return training sample
        # ----------------------------------------------------

        return {
            "pixel_values": (
                inputs.pixel_values.squeeze(0)
            ),

            "image_grid_thw": (
                inputs.image_grid_thw.squeeze(0)
            ),

            "input_ids": (
                inputs.input_ids.squeeze(0)
            ),

            "attention_mask": (
                inputs.attention_mask.squeeze(0)
            ),

            "labels": (
                labels.squeeze(0)
            ),
        }

    except Exception as e:

        print(
            f"❌ Error while processing sample: {e}"
        )

        return None


# ============================================================
# CUSTOM COLLATOR
# ============================================================

def custom_collator(features):
    """
    Collate variable-length multimodal samples
    into a training batch.
    """

    # --------------------------------------------------------
    # Input IDs
    # --------------------------------------------------------

    input_ids = pad_sequence(
        [
            f["input_ids"]
            for f in features
        ],
        batch_first=True,
        padding_value=(
            processor.tokenizer.pad_token_id
        ),
    )

    # --------------------------------------------------------
    # Attention mask
    # --------------------------------------------------------

    attention_mask = pad_sequence(
        [
            f["attention_mask"]
            for f in features
        ],
        batch_first=True,
        padding_value=0,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = pad_sequence(
        [
            f["labels"]
            for f in features
        ],
        batch_first=True,
        padding_value=-100,
    )

    # --------------------------------------------------------
    # Return batch
    # --------------------------------------------------------

    return {
        "input_ids": input_ids,

        "attention_mask": attention_mask,

        "labels": labels,

        "image_grid_thw": torch.stack(
            [
                f["image_grid_thw"]
                for f in features
            ]
        ),

        "pixel_values": torch.cat(
            [
                f["pixel_values"]
                for f in features
            ],
            dim=0,
        ),
    }


# ============================================================
# DATASET
# ============================================================

class ArabicOCRDataset(Dataset):
    """
    Dataset for Arabic manuscript OCR.

    Each CSV row should contain:

        file_name
        text

    Optional:

        type
    """

    def __init__(self, df):

        self.df = (
            df.reset_index(drop=True)
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        file_name = str(
            row["file_name"]
        )

        text = str(
            row["text"]
        )

        image_path = find_image(
            file_name
        )

        # ----------------------------------------------------
        # Missing image
        # ----------------------------------------------------

        if image_path is None:

            print(
                f"⚠️ Image not found: {file_name}"
            )

            # Try the next sample.
            return self.__getitem__(
                (idx + 1) % len(self.df)
            )

        # ----------------------------------------------------
        # Preprocess sample
        # ----------------------------------------------------

        result = preprocess_sample(
            image_path,
            text,
        )

        # ----------------------------------------------------
        # Failed preprocessing
        # ----------------------------------------------------

        if result is None:

            return self.__getitem__(
                (idx + 1) % len(self.df)
            )

        return result


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

train_df = pd.read_csv(
    CSV_TRAIN
)

val_df = pd.read_csv(
    CSV_VAL
)


print(
    f"\nTrain samples: {len(train_df)}"
)

if "type" in train_df.columns:

    print(
        train_df["type"].value_counts()
    )


print(
    f"\nValidation samples: {len(val_df)}"
)

if "type" in val_df.columns:

    print(
        val_df["type"].value_counts()
    )


# ------------------------------------------------------------
# Create datasets
# ------------------------------------------------------------

train_dataset = ArabicOCRDataset(
    train_df
)

eval_dataset = ArabicOCRDataset(
    val_df
)


print(
    f"\n✅ Train: {len(train_dataset)} | "
    f"Validation: {len(eval_dataset)}"
)


# ============================================================
# TRAINING CALLBACK
# ============================================================

class ProgressCallback(TrainerCallback):

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs,
    ):

        if logs:

            print(
                f"Step {state.global_step} | "
                f"Epoch {round(state.epoch or 0, 3)} | "
                f"Loss: {logs.get('loss', 'N/A')} | "
                f"Eval: {logs.get('eval_loss', '')}"
            )


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = Seq2SeqTrainingArguments(

    output_dir=OUTPUT_DIR,

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRAD_ACCUM,

    # --------------------------------------------------------
    # Training duration
    # --------------------------------------------------------

    num_train_epochs=EPOCHS,

    learning_rate=LR,

    # --------------------------------------------------------
    # Learning rate scheduler
    # --------------------------------------------------------

    lr_scheduler_type="cosine",

    warmup_steps=500,

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    bf16=True,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=5,

    # --------------------------------------------------------
    # Evaluation / checkpoints
    # --------------------------------------------------------

    eval_steps=500,

    save_steps=500,

    eval_strategy="steps",

    save_strategy="steps",

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    # --------------------------------------------------------
    # Memory / dataloader
    # --------------------------------------------------------

    gradient_checkpointing=False,

    dataloader_num_workers=2,

    remove_unused_columns=False,

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------

    report_to="none",

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optim="adafactor",

    # --------------------------------------------------------
    # Checkpoint management
    # --------------------------------------------------------

    save_total_limit=3,

    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    max_grad_norm=1.0,

    weight_decay=0.01,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Seq2SeqTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    data_collator=custom_collator,

    processing_class=processor,

    callbacks=[
        ProgressCallback()
    ],
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("🚀 Starting LoRA fine-tuning...")
print("=" * 70)
print()


trainer.train()


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("Saving fine-tuned model...")


trainer.save_model(
    OUTPUT_DIR
)

processor.save_pretrained(
    OUTPUT_DIR
)


print()
print("=" * 70)
print("Fine-tuning completed!")
print(f"Model saved to: {OUTPUT_DIR}")
print("=" * 70)
