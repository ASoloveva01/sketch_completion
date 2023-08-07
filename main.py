from models.cgans import Generator, Discriminator
from models.classifier import SketchANet
import torch
import training
from data_customizer import train_dir, test_dir, customize_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import CreateDataset

classes = 150

#customize_dataset(num_classes=classes)

transforms_with_erasing = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(p=1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=1)

])

transforms_wout_erasing = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))

])

train_data = CreateDataset(targ_dir=train_dir, mask_transform=transforms_with_erasing,
                           real_transform=transforms_wout_erasing)

test_data = CreateDataset(targ_dir=test_dir, mask_transform=transforms_with_erasing,
                           real_transform=transforms_wout_erasing)


train_dataloader = DataLoader(dataset=train_data,
                                     batch_size=20,
                                     num_workers=0,
                                     shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                                    batch_size=20,
                                    num_workers=0,
                                    shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = Generator().to(device)
discriminator = Discriminator().to(device)
classifier = SketchANet(num_classes=classes).to(device)
G_optimizer = torch.optim.Adam(generator.parameters(), lr=0.01)
D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.01)
epochs = 100
'''
training1.train_loop(generator, discriminator, classifier, G_optimizer, D_optimizer, train_dataloader,
                     test_dataloader,epochs)
MODEL_PATH = Path("pretrained_models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "generator.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(obj=generator.state_dict(),
           f=MODEL_SAVE_PATH)
'''
mask_img, _, _ = next(iter(test_dataloader))
print(mask_img)

