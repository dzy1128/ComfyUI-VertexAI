from google import genai
from google.genai import types
import os

def generate():
  client = genai.Client(
      vertexai=True,
      api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"),
  )

  # 读取本地图片
  with open("./assets/pikachu.png", "rb") as f:
      image_bytes = f.read()
  
#   with open("./assets/girl1.png", "rb") as f:
#       image_bytes1 = f.read()

  msg1_image1 = types.Part.from_bytes(
      data=image_bytes,
      mime_type="image/png",
  )
#   msg1_image2 = types.Part.from_bytes(
#       data=image_bytes1,
#       mime_type="image/png",
#   )

  model = "gemini-3-pro-image-preview"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_image1,
        #msg1_image2,
        types.Part.from_text(text="""Make it transform into a Super Saiyan.""")
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 32768,
    response_modalities = ["TEXT", "IMAGE"],
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    image_config=types.ImageConfig(
      aspect_ratio="1:1",
      image_size="1K",
      output_mime_type="image/png",
    ),
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config, 
    ):
    # 检查是否有 candidates
    if chunk.candidates:
      for candidate in chunk.candidates:
        if candidate.content and candidate.content.parts:
          for part in candidate.content.parts:
            # 打印思考过程文本
            if hasattr(part, 'text') and part.text:
              print(part.text)
            
            # 获取并保存图片数据
            if hasattr(part, 'inline_data') and part.inline_data:
              image_data = part.inline_data.data
              mime_type = part.inline_data.mime_type
              
              # 保存图片到本地
              output_path = "./assets/generated_image.png"
              with open(output_path, "wb") as f:
                f.write(image_data)
              print(f"\n✅ 图片已保存到: {output_path}")
              print(f"MIME类型: {mime_type}")

generate()