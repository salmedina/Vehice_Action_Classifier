import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

class Img2Vec(object):
    def __init__(self):
        # Load the pretrained model
        self.model = models.resnet18(pretrained=True)
        # Use the model object to select the desired layer
        self.layer = self.model._modules.get('avgpool')
        self.model.eval()

        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()


    def get_embedding(self, img):
        # 2. Create a PyTorch Variable with the transformed image
        t_img = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0))
        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        output_embedding = torch.zeros(512)

        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            output_embedding.copy_(o.squeeze())

        # 5. Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        self.model(t_img)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector
        return output_embedding

if __name__ == '__main__':
    img2vec = Img2Vec()
    img_path = '/mnt/Alfheim/Data/MEVA/DirectionFrames/batch04/clean/2018-03-15_15-55-07_16-00-07_hospital_G436__f92c856aefaf14069f02ac022ae44b8a69515048_b2efe8ebaccf7f71266c210e501b9f31447eedfb__Vehicle_U-Turn_f_795.jpg'
    img = Image.open(img_path)
    img_vec = img2vec.get_embedding(img)
    print(img_vec)
    print(type(img_vec))