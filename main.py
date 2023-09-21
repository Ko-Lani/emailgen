import openai
import torch
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained(
    "skt/kogpt2-base-v2",
)
model.eval()

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    eos_token="</s>",
)

# 카테고리에 따라 양식 선택
def select_template(category):
    if category == "course":
        return "course.txt"
    elif category == "lab":
        return "lab.txt"
    else:
        return "else.txt"

# GPT-3를 통해 세부사항 생성
def generate_details(category):
    input_ids = tokenizer.encode(f'다음의 말을 메세지를 한문장으로 쓸거야.{category}', return_tensors="pt")
    input_ids = tokenizer.encode(f'다음의 말을 메세지를 한문장으로 쓸거야.수업', return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=10,
            max_length=50,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            temperature=0.9,
            top_k=50,
            top_p=0.92,
        )
    return tokenizer.decode([el.item() for el in generated_ids[0]])

# 이메일 내용 생성
def generate_email_content(category):
    template_file = select_template(category)
    with open(template_file, "r") as file:
        template = file.read()

    details = generate_details(category)
    template_with_details = template.replace("{{details}}", details)

    return template_with_details

if __name__ == "__main__":
    category = input("카테고리를 입력하세요 (course, lab 등): ")
    email_content = generate_email_content(category)
    print("이메일 내용:")
    print(email_content)
