import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_loop(generator, discriminator, classifier,
               G_optimizer, D_optimizer, train_loader,
               test_loader, num_epochs=20, device=device,
               plot=False):
    # Инициализируем функции потерь
    gen_adv_loss_fn = nn.BCEWithLogitsLoss()
    pixel_loss_fn = nn.L1Loss()
    classifier_loss_fn = nn.CrossEntropyLoss()

    D_train_loss_list, G_train_loss_list, D_test_loss_list, G_test_loss_list = [], [], [], []
    for epoch in tqdm(range(num_epochs)):

        D_train_loss, G_train_loss, D_test_loss, G_test_loss = 0, 0, 0, 0

        generator.train()
        discriminator.train()
        classifier.train()

        ### Обучение ###
        for index, (mask_images, real_images, labels) in enumerate(train_loader):

            # Отправляем изображения на соответсвующее устройство
            mask_images = mask_images.to(device)
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Инициализируем метки с 1 и 0
            real_target = Variable(torch.ones(real_images.size(0)).to(device))
            fake_target = Variable(torch.zeros(real_images.size(0)).to(device))

            #### Обучение дискриминатора ###
            D_optimizer.zero_grad()

            # Измеряем способность дискриминатора отличать реальные изображения от фейковых
            D_real_loss = gen_adv_loss_fn(discriminator((mask_images, real_images)).reshape(-1), real_target)
            gen_images = generator(mask_images).to(device)
            D_fake_loss = gen_adv_loss_fn(discriminator((mask_images, gen_images)).reshape(-1), fake_target)
            D_loss = (D_real_loss + D_fake_loss) / 2
            D_train_loss += D_loss

            D_loss.backward(retain_graph=True)
            D_optimizer.step()

            #### Обучение генератора ###
            G_optimizer.zero_grad()
            classifier.zero_grad()

            # Вычисляем оригинальный лосс gans, расстояние l1 между фейковыми и реальными изображениями
            # и кросс-энтропию классификатора
            G_adv_loss = gen_adv_loss_fn(discriminator((mask_images, gen_images)).reshape(-1), real_target)
            G_pix_loss = pixel_loss_fn(gen_images, real_images)

            C_loss = classifier_loss_fn(classifier(gen_images).to(device), labels)

            # Вычисляем лосс генератора находя взвешенную сумму 3 лоссов
            G_loss = G_adv_loss + 0.5 * C_loss + 100 * G_pix_loss

            G_train_loss += G_loss
            G_loss.backward()
            G_optimizer.step()

        D_train_loss_list.append(D_train_loss)
        G_train_loss_list.append(G_train_loss)

        generator.eval()
        discriminator.eval()
        classifier.eval()

        ### Валидация ###
        with torch.inference_mode():
            for index, (mask_images, real_images, labels) in enumerate(test_loader):
                mask_images = mask_images.to(device)
                real_images = real_images.to(device)
                labels = labels.to(device)

                real_target = Variable(torch.ones(real_images.size(0)).to(device))
                fake_target = Variable(torch.zeros(real_images.size(0)).to(device))

                D_real_loss = gen_adv_loss_fn(discriminator((mask_images, real_images)).reshape(-1), real_target)
                gen_images = generator(mask_images).to(device)
                D_fake_loss = gen_adv_loss_fn(discriminator((mask_images, gen_images)).reshape(-1), fake_target)
                D_loss = (D_real_loss + D_fake_loss) / 2
                D_test_loss += D_loss

                G_adv_loss = gen_adv_loss_fn(discriminator((mask_images, gen_images)).reshape(-1), real_target)
                G_pix_loss = pixel_loss_fn(gen_images, real_images)

                C_loss = classifier_loss_fn(classifier(gen_images).to(device), labels)

                G_loss = G_adv_loss + 0.5 * C_loss + 100 * G_pix_loss

                G_test_loss += G_loss

            D_test_loss_list.append(D_test_loss)
            G_test_loss_list.append(G_test_loss)
    if plot:
        plot_loss_curves({"disc_train_loss":D_train_loss_list, "gen_train_loss":G_train_loss_list,
                          "disc_test_loss":D_test_loss_list,"gen_test_loss":G_test_loss_list})

def plot_loss_curves(results: Dict[str, List[float]]):
  
    D_train_loss = results['disc_train_loss']
    G_train_loss = results['gen_train_loss']

    D_test_loss = results['disc_test_loss']
    G_test_loss = results['gen_test_loss']

    epochs = range(len(results['disc_train_loss']))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, D_train_loss, label='disc_loss')
    plt.plot(epochs, G_train_loss, label='gen_loss')
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, D_test_loss, label='disc_loss')
    plt.plot(epochs, G_test_loss, label='gen_loss')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.legend()

