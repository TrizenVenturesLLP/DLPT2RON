

from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import torch
import gradio as gr

# Set device for model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load language model
language_model_name = "Qwen/Qwen2-1.5B-Instruct"
language_model = AutoModelForCausalLM.from_pretrained(
    language_model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(language_model_name)

def process_input(input_text, action):
    if action == "Translate to English":
        prompt = f"Please translate the following text into English：{input_text}"
        lang = "en"
    elif action == "Translate to Chinese":
        prompt = f"Please translate the following text into Chinese：{input_text}"
        lang = "zh-cn"
    elif action == "Translate to Japanese":
        prompt = f"Please translate the following text into Japanese：{input_text}"
        lang = "ja"
    else:
        prompt = input_text
        lang = "en"

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = language_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text, lang

def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang)
    filename = "src/videos/output_audio.mp3"
    tts.save(filename)
    return filename

def handle_interaction(input_text, action):
    output_text, lang = process_input(input_text, action)
    audio_filename = text_to_speech(output_text, lang)
    return output_text, audio_filename

# Gradio interface options
action_options = ["Translate to English", "Translate to Chinese", "Translate to Japanese", "Chat"]

iface = gr.Interface(
    fn=handle_interaction,
    inputs=[
        gr.Textbox(label="Input Text"),
        gr.Dropdown(action_options, label="Select Action")
    ],
    outputs=[
        gr.Textbox(label="Output Text"),
        gr.Audio(label="Output Audio")
    ],
    title="Translation and Chat App using AI",
    description="Translate input text or chat based on the selected action.",
    theme="gradio/soft"
)

if __name__ == "__main__":
    iface.launch(share=True)

