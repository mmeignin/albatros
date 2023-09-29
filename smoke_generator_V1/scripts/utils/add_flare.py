from Lens_flare.data_loader import Flare_Image_Loader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def add_lens_flare(image_path,flare_path):
        transform_base=transforms.Compose([transforms.Resize((1080,1080))])

        transform_flare=transforms.Compose([transforms.RandomAffine(degrees=(0,360),scale=(0.8,1.5),translate=(300/1440,300/1440),shear=(-20,20)),
                                transforms.CenterCrop((1080,1080)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip()
                                ])

        flare_image_loader=Flare_Image_Loader(image_path,transform_base,transform_flare)
        flare_image_loader.load_scattering_flare(flare_path,flare_path+'/Scattering_Flare/Streak')
        flare_image_loader.load_reflective_flare(flare_path,flare_path+'/Reflective_Flare')
        __,__,test_merge_img,__=flare_image_loader[0]
        return test_merge_img 

image_path=r'D:\mploi\Documents\Albatros\albatros\smoke_dataset_V1\images'
flare_path=r'D:\mploi\Documents\Albatros\albatros\smoke_generator_V1\scripts\utils\Lens_flare\Flare7Kpp\Flare7K'
add_lens_flare(image_path,flare_path)