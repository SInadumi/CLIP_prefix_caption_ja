import os
import json
import torch
import pickle
import scripts.model.japanese_clip as ja_clip
from PIL import Image
from tqdm import tqdm
from transformers import T5Tokenizer
from load_gazevqa import get_coco_path


def main():
    # config tokenizer, clip, STAIR dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_path = f"{os.environ['STAIR_CAPTION_DIR']}/stair_captions_v1.2_train.json"
    out_path = f"{os.environ['STAIR_CAPTION_DIR']}/stair_split_ViT-B-16_train"
    coco_dir = f"{os.environ['COCO_DIR']}"
    model, preprocess = ja_clip.load(
        "rinna/japanese-clip-vit-b-16",
        cache_dir="/tmp/japanese_clip",
        device=device
    )
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
    data = json.load(open(input_path, 'r', encoding="utf-8"))
    print("%0d captions loaded from json " % len(data["annotations"]))

    all_embeddings = []
    all_captions = []

    ids = 0
    for _ in tqdm(range(len(data["annotations"])), disable=True):
        d = data["annotations"][ids]

        # encode mscoco image
        img_id = d["image_id"]
        image = Image.open(get_coco_path(coco_dir, img_id))
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = model.get_image_features(image).cpu()
        d["clip_embedding"] = ids
        # tokenize caption text (using T5Tokenizer)
        d["caption"] = tokenizer.tokenize(d["caption"])
        all_embeddings.append(prefix)
        all_captions.append(d)

        # save embeddings and caption per 10000 times
        if (ids + 1) % 10000 == 0:
            with open(out_path + f"_{ids+1}.pkl", 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
                # メモリ開放
                del all_embeddings
                del all_captions
                all_embeddings = []
                all_captions = []
        ids += 1

    with open(out_path + f"_{ids+1}.pkl", 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    main()
