import argparse
import image_classification_utils

parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help='Image path to predict')
parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth', help='Checkpoint load file. default=./checkpoint.pth')
parser.add_argument('--cat_file', type=str, default='cat_to_name.json', help='Categories mapping. default=cat_to_name.json')
parser.add_argument('--topk', type=int, default=5, help='Number of most likely classes to return. dafault=5')
parser.add_argument('--gpu', action="store_true", help='Use GPU')
args = parser.parse_args()

device = image_classification_utils.select_device(args.gpu)

model, _, _, _ = image_classification_utils.load_checkpoint(args.checkpoint, device)
categories = image_classification_utils.load_categories(args.cat_file)

prob, _, class_names = image_classification_utils.predict(model, args.image_path, categories, args.topk, device)

image_classification_utils.print_prediction(prob, class_names)
