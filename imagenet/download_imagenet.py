from pathlib import Path
import utils
import requests
import os
#print(os.getcwd())

username, accesskey = utils.get_username_and_password("imagenet/credentials")

s = requests.Session()
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"
s.headers.update({'user-agent': user_agent})

def save_file(url, output_file):
    page = s.get(url)
    with open(output_file, "wb") as f:
        f.write(page.content)

image_type="dogs"
Path(f"imagenet/images/{image_type}").mkdir(exist_ok=True, parents=True)
for i,url in enumerate(Path(f"imagenet/{image_type}").open().readlines()):
    fname = Path(url).name.strip()
    save_file(url, f"imagenet/images/{image_type}/{fname}")
    if i > 20:
        break