import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import inception_v3


net = inception_v3(pretrained=True).cuda()


def inception_score(images, batch_size=5):
    scores = []
    for i in range(int(math.ceil(float(len(images)) / float(batch_size)))):
        batch = Variable(torch.cat(images[i * batch_size: (i + 1) * batch_size], 0))
        s, _ = net(batch)  # skipping aux logits
        scores.append(s)
    p_yx = F.softmax(torch.cat(scores, 0), 1)
    p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
    KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
    final_score = KL_d.mean()
    return final_score



if __name__=='__main__':
    if softmax is None:
      _init_inception()
      
    def get_images(filename):
        return scipy.misc.imread(filename)


    for iterations in range(100, 5000, 100):
        images = []
        for temp in ['a', 'b']:
            print(iterations)
            filenames = 'out_*_{}{}_*_.jpg'.format(iterations, temp)
            filenames = os.path.join('celebA_results/for_inception/', filenames)
            filenames = glob.glob(filenames)
            print(filenames)
            images.extend([get_images(filename) for filename in filenames])
            #print(len(images))
        print(inception_score(images))