#Loss_D: 0.153982 0.164330 Loss_G: 0.213881 0.208388 loss_cyc 0.047689 818.7505164146423

import matplotlib.pyplot as plt
import re
import numpy as np

REGEX_LINE = re.compile(r'Loss_D\: (?P<loss_D_1>\d+\.\d+) (?P<loss_D_2>\d+\.\d+) Loss_G\: (?P<loss_G_1>\d+\.\d+) (?P<loss_G_2>\d+\.\d+) loss_cyc (?P<loss_cyc_1>\d+\.\d+) (?P<loss_cyc_2>\d+\.\d+)')

#names = ['loss_D_1', 'loss_D_2', 'loss_G_1', 'loss_G_2', 'loss_cyc_1', 'loss_cyc_2']
names = ['loss_D_1', 'loss_D_2', 'loss_G_1', 'loss_G_2', 'loss_cyc_1', 'loss_cyc_2']
useful_names = names[:-1]
disp_names = ['FW-Discriminator', 'BW-Discriminator', 'FW-Generator', 'BW-Generator', 'Combined GAN']

filename = 'disguise_loss.txt'
results = [[] for i in range(6)]
with open(filename, 'r') as f:
    lines = f.readlines()

    for l in lines:
        found_dict = REGEX_LINE.search(l)
        if found_dict is not None:
            found_dict = found_dict.groupdict()
            for f in found_dict:
                results[names.index(f)].append(float(found_dict[f]))
        

results = np.array(results)
print(results)        

plt.figure(figsize=(10,8))

for i in range(len(useful_names)):
    plt.semilogy(results[i], label=disp_names[i])

plt.title('CycleGAN Disguise Addition/Removal Losses', fontsize=20)
plt.ylabel('Loss', fontsize=18)
plt.xlabel('Number of Batches', fontsize=18)
plt.legend(fontsize=16)
plt.savefig('cyclegan_loss.png')
#plt.show()