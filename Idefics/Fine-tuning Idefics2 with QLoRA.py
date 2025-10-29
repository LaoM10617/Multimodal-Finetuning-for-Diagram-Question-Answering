# Reference: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Idefics2/Fine_tune_Idefics2_for_multi_page_PDF_question_answering_on_DUDE.ipynb

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
# from transformers import IdeficsForVisionText2Text, AutoProcessor
from transformers import BitsAndBytesConfig, Idefics2ForConditionalGeneration, AutoProcessor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from torch.utils.data import DataLoader
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

MAX_LENGTH = 1024
WANDB_PROJECT = "Finetuned-Idefics2"
WANDB_NAME = "demo-run"

def resize_image(image, resize_factor: float = 0.8):
    width, height = image.size
    image = image.resize((int(resize_factor * width), int(resize_factor * height)))
    return image

def display_image(image, resize_factor: float = 0.8):
    image = resize_image(image, resize_factor)
    plt.imshow(image)
    plt.show()

class Idefics2Dataset(Dataset):
    """
    PyTorch Dataset for Idefics2. This class takes a HuggingFace Dataset as input.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns one item of the dataset.

        Returns:
            question : the question text
            options : the available options
            answer : the correct answer
            image : the image associated with the question
        """
        samples = self.dataset
        question = samples['question'][idx]
        options = samples['options'][idx]
        answer = samples["answer"][idx]
        image = samples["image"][idx]

        return question, options, answer, image


dataset = load_dataset('lmms-lab/ai2d')

print(dataset['test']['question'][0])

split_dataset = dataset['test'].train_test_split(test_size=0.1)

# train_dataset = Idefics2Dataset(split_dataset['train'][:10])
# validation_dataset = Idefics2Dataset(split_dataset['test'][:10])
train_dataset = Idefics2Dataset(split_dataset['train'])
validation_dataset = Idefics2Dataset(split_dataset['test'])
print(len(train_dataset))
print(len(validation_dataset))

# Load the IDEFICS model (use the tiny model for try)
checkpoint = "HuggingFaceM4/idefics2-8b"
# checkpoint = "HuggingFaceM4/idefics-80b"

# most memory friendly version
# USE_QLORA = True

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.bfloat16,
        # device_map='auto',
        quantization_config=bnb_config,
    )

'''
# Full fine-tuning
model = Idefics2ForConditionalGeneration.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,
    device_map=None,
    _attn_implementation="flash_attention_2",
    )
'''

lora_config = LoraConfig(
          r=8,
          lora_alpha=8,
          lora_dropout=0.1,
          target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
          use_dora=False,
          init_lora_weights="gaussian",
      )

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

'''
# In this case, you only have to load the model without moving to CUDA.
# As exposed in the error log, when the parameter device_map is 'auto', device_map='auto',
# and you are using a quantization configuration, bnbconfig of 4 bits or 8 bits, the model is automatically moved to CUDA.
# Hence, it could not be moved twice and that error raises.
device = torch.device("cuda")
model.to(device)
'''

processor = AutoProcessor.from_pretrained(checkpoint)
image_token_id = processor.tokenizer.additional_special_tokens_ids[processor.tokenizer.additional_special_tokens.index("<image>")]

def train_collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        question, options, answer, image_example = example
        question += 'Based on the given image, please answer the question, the answer should be a number. \
        0 represents the first option. 1 represents the second option. 2 represents the third option. \
        3 represents the fourth option. Please choose from the following options using a number above: '
        content = [{"type": "image"}]
        content += [{"type": "text", "text": question}]
        content += [{"type": "text", "text": options}]

        # Create inputs
        messages = [
            {
                "role": "user",
                "content": content,
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ]
            },
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        #print(f"Prompt:{prompt}")
        #print(f"Image:{image_example}")
        texts.append(prompt)
        images.append(image_example)

    print(f"Number of texts: {len(texts)}, Number of images: {len(images)}")
    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    print(batch)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == model.config.image_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    pixel_attention_mask = batch["pixel_attention_mask"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, pixel_attention_mask, labels

train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=2, shuffle=True)
batch = next(iter(train_dataloader))
input_ids, attention_mask, pixel_values, pixel_attention_mask, labels = batch
processor.batch_decode(input_ids)
print(pixel_values.shape)

def eval_collate_fn(examples):
    # Feed the prompt to the model
    images = []
    texts = [] # Prompts
    answers = []
    for example in examples:
        question, options, answer, image_example = example
        question += 'Based on the given image, please answer the question, the answer should be a number. \
                0 represents the first option. 1 represents the second option. 2 represents the third option. \
                3 represents the fourth option. Please choose from the following options using a number above: '
        content = [{"type": "image"}]
        content += [{"type": "text", "text": question}]
        content += [{"type": "text", "text": options}]

        messages = [
            {
                "role": "user",
                "content": content,
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images.append(image_example)
        texts.append(text.strip())
        answers.append(answer)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    pixel_attention_mask = batch["pixel_attention_mask"]

    return input_ids, attention_mask, pixel_values, pixel_attention_mask, answers


class Idefics2ModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, pixel_attention_mask, labels = batch

        outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                pixel_attention_mask=pixel_attention_mask,
                                labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, pixel_attention_mask, answers = batch

        # autoregressively generate token IDs
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask,
                                       max_new_tokens=MAX_LENGTH)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        total = 0
        correct = 0
        for pred, answer in zip(predictions, answers):
            if pred == answer:
                correct += 1
            total += 1
        return correct / total

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    # Remove 'num_workers' to disable parallel loads to avoid the error 'DataLoader exited unexpectedly'
    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(validation_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False)

config = {"max_epochs": 10,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 1,
          "precision": "16-mixed", # we'll use mixed precision
          # "seed":2022, # can be used for reproducibility
          "warmup_steps": 50,
          "result_path": "./result",
          "verbose": True
}
model_module = Idefics2ModelPLModule(config, processor, model)

early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode='min')

wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.get("max_epochs"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        precision=config.get("precision"),
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=early_stop_callback
)

trainer.fit(model_module)