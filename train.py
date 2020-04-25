import argparse
import image_classification_utils

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, default='flowers/', help='Dataset directory. default=flowers/')
parser.add_argument('--save_file', type=str, default='./checkpoint.pth', help='Checkpoint save file. default=./checkpoint.pth')
parser.add_argument('--type', choices=['resnet18', 'alexnet', 'vgg16', 'squeezenet', 'densenet'], default='vgg16', help='Model architectures. default=vgg16')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate. default=0.003')
parser.add_argument('--hidden_layers_outputs', type=int, nargs='+', default=[2048, 512], help='Hidden layers outputs. default=2048 512')
parser.add_argument('--hidden_layers_dropout', type=int, nargs='+', default=[0.5, 0.5], help='Hidden layers dropout. default=0.5 0.5')
parser.add_argument('--hidden_layers_af', choices=['ReLU', 'Sigmoid', 'Tanh'], nargs='+', default=['ReLU', 'ReLU'], help='Hidden layers activation function. default=ReLU ReLU')
parser.add_argument('--outputs', type=int, default=102, help='Number of model outputs. default=102')
parser.add_argument('--epochs', type=int, default=16, help='Number of epochs. default=16')
parser.add_argument('--print_interval', type=int, default=50, help='Print interval. 0 to not print. default=50')
parser.add_argument('--gpu', action="store_true", help='Use GPU')
args = parser.parse_args()

device = image_classification_utils.select_device(args.gpu)
hidden_layers = image_classification_utils.create_hidden_layers(args.hidden_layers_outputs, args.hidden_layers_dropout, args.hidden_layers_af)
output = image_classification_utils.create_output(args.outputs)
dataloaders, class_to_idx = image_classification_utils.load_data(args.data_dir)

model = image_classification_utils.create_model(device, class_to_idx, args.type, hidden_layers, output)

optimizer = image_classification_utils.create_optimizer(model, args.learning_rate)

image_classification_utils.do_deep_learning(model, optimizer, dataloaders['train'], dataloaders['valid'], args.epochs, args.print_interval, device)

image_classification_utils.save_checkpoint(model, optimizer, args.epochs, args.learning_rate, args.save_file)
