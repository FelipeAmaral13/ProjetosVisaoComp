import os
import glob
import random
from vit_finetune import ViTFinetune
import matplotlib.pyplot as plt

data_dir = os.path.join('kaggle\treino')
val_dir = os.path.join('kaggle\valid')

batch_size = 10
num_epochs = 10
learning_rate = 2e-5

model = ViTFinetune(num_classes=2)  # Substitua 2 pelo número de classes do seu problema
model.set_device('cuda')

# Carregar o dataset de treinamento
train_loader, num_classes = model.load_dataset(data_dir, batch_size)

# Carregar o dataset de validação
val_loader = model.load_validation_dataset(val_dir, batch_size)

# Treinar o modelo
model.train_model(train_loader, val_loader, num_epochs, learning_rate)

# Salvar o modelo treinado em um arquivo
model.save_model(os.path.join('modelos','modelo_treinado.pth'))

# Carregar o modelo treinado a partir do arquivo
model_carregado = ViTFinetune(num_classes=2)
model_carregado.load_model(os.path.join('modelos','modelo_treinado.pth'))


# Fazer a predição da imagem
image_path = glob.glob(os.path.join('pred', '*.jpg'))
selected_images = random.sample(image_path, 8)

fig, axs = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axs.flat):
    image_path = selected_images[i]
    predicted_class, probabilities, image = model_carregado.predict_image(image_path)
    ax.set_title(f"Predict Class: {predicted_class}")
    ax.imshow(image)
    
    ax.axis('off')

plt.tight_layout()
plt.show()

