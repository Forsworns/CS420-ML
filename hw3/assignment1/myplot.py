import matplotlib.pyplot as plt
from sl_rm import load_result
from configs import *
import os

if __name__ == "__main__":
    labels = dict()
    contents = dict()
    os.chdir('results')
    for _, dirs, _ in os.walk("./", topdown=True):
        for method in dirs:
            labels.update({method: []})
            contents.update({method: []})
            for _, _, files in os.walk(method):
                '''
                # rename the files
                for _, _, fs in os.walk('.'):
                    for f in fs:
                        if 'decision_function_shape' in f:
                            paras = f.split('decision_function_shape')
                            new_file = paras[0]+'decision-function-shape'+paras[1]
                            os.rename(f, new_file)

                for _, _, fs in os.walk('.'):
                    for f in fs:
                        if 'hidden_layer_sizes' in f:
                            paras = f.split("'hidden-layer-sizes',")
                            paras[1] = paras[1].replace(',','-')
                            new_file = paras[0]+"'hidden-layer-sizes',"+paras[1]
                            os.rename(f, new_file)
                '''
                for f in files:
                    paras = f[0:-4].split('_')
                    labels[method].append({})
                    for para in paras:
                        para = para.strip('()')
                        key, value = para.split(',')
                        key = key.strip("'")
                        value = value.strip(" '()")
                        print(key,value)
                        labels[method][-1].update({key: value})
                    content = load_result(model_name=method, file_name=f)
                    contents[method].append(content)
    print(labels)
    print(contents)

    # plot for a1 svm
    plt.figure(figsize=(3, 6))
    plt.subplots_adjust(bottom=.05, top=0.97, hspace=.25, wspace=.15,
                        left=.05, right=.99)
    scs_ovo = {x: [] for x in KERNELS}
    scs_ovr = {x: [] for x in KERNELS}
    print(labels[A1_SVM])
    for idx, paras in enumerate(labels[A1_SVM]):
        kernel = paras['kernel']
        sc = contents[A1_SVM][idx]['score']
        if paras['decision-function-shape'] == 'ovo':
            scs_ovo[kernel].append(sc)
        elif paras['decision-function-shape'] == 'ovr':
            scs_ovr[kernel].append(sc)
    print(scs_ovo)
    for kernel in KERNELS:
        color = COLORS[KERNELS_MAP[kernel]]
        plt.subplot(1, 2, 1)
        plt.plot(CS, scs_ovo[kernel], color=color, label=kernel)
        plt.title('ovo score')
        plt.ylabel("score")
        plt.xlabel("C")
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(fontsize='xx-large')
        plt.subplot(1, 2, 2)
        plt.plot(CS, scs_ovr[kernel], color=color, label=kernel)
        plt.title('ovr score')
        plt.ylabel("score")
        plt.xlabel("C")
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(fontsize='xx-large')
    plt.show()

    # plot for a2 svm
    plt.figure(figsize=(3, 6))
    plt.subplots_adjust(bottom=.05, top=0.97, hspace=.25, wspace=.15,
                        left=.05, right=.99)
    scs_ovo = {x: [] for x in KERNELS}
    scs_ovr = {x: [] for x in KERNELS}
    print(labels[A2_SVM])
    for idx, paras in enumerate(labels[A2_SVM]):
        kernel = paras['kernel']
        sc = contents[A2_SVM][idx]['score']
        if paras['decision-function-shape'] == 'ovo':
            scs_ovo[kernel].append(sc)
        elif paras['decision-function-shape'] == 'ovr':
            scs_ovr[kernel].append(sc)
    print(scs_ovo)
    for kernel in KERNELS:
        color = COLORS[KERNELS_MAP[kernel]]
        plt.subplot(1, 2, 1)
        plt.plot(CS, scs_ovo[kernel], color=color, label=kernel)
        plt.title('ovo score')
        plt.ylabel("score")
        plt.xlabel("C")
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(fontsize='xx-large')
        plt.subplot(1, 2, 2)
        plt.plot(CS, scs_ovr[kernel], color=color, label=kernel)
        plt.title('ovr score')
        plt.ylabel("score")
        plt.xlabel("C")
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(fontsize='xx-large')
    plt.show()

    # plot for b svm
    plt.figure(figsize=(3, 6))
    plt.subplots_adjust(bottom=.05, top=0.97, hspace=.25, wspace=.15,
                        left=.05, right=.99)
    scs_ovo = {x: [] for x in KERNELS}
    scs_ovr = {x: [] for x in KERNELS}
    print(labels[B_SVM])
    for idx, paras in enumerate(labels[B_SVM]):
        kernel = paras['kernel']
        sc = contents[B_SVM][idx]['score']
        if paras['decision-function-shape'] == 'ovo':
            scs_ovo[kernel].append(sc)
        elif paras['decision-function-shape'] == 'ovr':
            scs_ovr[kernel].append(sc)
    print(scs_ovo)
    for kernel in KERNELS:
        color = COLORS[KERNELS_MAP[kernel]]
        plt.subplot(1, 2, 1)
        plt.plot(CS, scs_ovo[kernel], color=color, label=kernel)
        plt.title('ovo score')
        plt.ylabel("score")
        plt.xlabel("C")
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(fontsize='xx-large')
        plt.subplot(1, 2, 2)
        plt.plot(CS, scs_ovr[kernel], color=color, label=kernel)
        plt.title('ovr score')
        plt.ylabel("score")
        plt.xlabel("C")
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(fontsize='xx-large')
    plt.show()


    # plot for a1 nn
    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(bottom=.05, top=0.97, hspace=.25, wspace=.15,
                        left=.05, right=.99)
    scs_relu = {x: [] for x in HIDDEN_P}
    scs_log = {x: [] for x in HIDDEN_P}
    scs_tan = {x: [] for x in HIDDEN_P}
    print(labels[A1_NN])
    for idx, paras in enumerate(labels[A1_NN]):
        layer = paras['hidden-layer-sizes']
        sc = contents[A1_NN][idx]['score']
        if paras['activation'] == 'relu':
            scs_relu[layer].append(sc)
        elif paras['activation'] == 'logistic':
            scs_log[layer].append(sc)
        elif paras['activation'] == 'tanh':
            scs_tan[layer].append(sc)
    print(scs_ovo)
    for layer in HIDDEN_P:
        color = COLORS[HIDDEN_MAP[layer]]
        plt.subplot(1, 3, 1)
        plt.plot(ALPHA, scs_relu[layer], color=color, label=layer.replace('-',','))
        plt.title('relu')
        plt.ylabel("score")
        plt.xlabel("alpha")
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(fontsize='xx-large')
        plt.subplot(1, 3, 2)
        plt.plot(ALPHA, scs_log[layer], color=color, label=layer.replace('-',','))
        plt.title('logistic')
        plt.ylabel("score")
        plt.xlabel("alpha")
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(fontsize='xx-large')
        plt.subplot(1, 3, 3)
        plt.plot(ALPHA, scs_tan[layer], color=color, label=layer.replace('-',','))
        plt.title('tanh')
        plt.ylabel("score")
        plt.xlabel("alpha")
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(fontsize='xx-large')
    plt.show()


    # plot for a2 nn
    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(bottom=.05, top=0.97, hspace=.25, wspace=.15,
                        left=.05, right=.99)
    scs_relu = {x: [] for x in HIDDEN_P}
    scs_log = {x: [] for x in HIDDEN_P}
    scs_tan = {x: [] for x in HIDDEN_P}
    print(labels[A2_NN])
    for idx, paras in enumerate(labels[A2_NN]):
        layer = paras['hidden-layer-sizes']
        sc = contents[A2_NN][idx]['score']
        if paras['activation'] == 'relu':
            scs_relu[layer].append(sc)
        elif paras['activation'] == 'logistic':
            scs_log[layer].append(sc)
        elif paras['activation'] == 'tanh':
            scs_tan[layer].append(sc)
    print(scs_ovo)
    for layer in HIDDEN_P:
        color = COLORS[HIDDEN_MAP[layer]]
        plt.subplot(1, 3, 1)
        plt.plot(ALPHA, scs_relu[layer], color=color, label=layer.replace('-',','))
        plt.title('relu')
        plt.ylabel("score")
        plt.xlabel("alpha")
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(fontsize='xx-large')
        plt.subplot(1, 3, 2)
        plt.plot(ALPHA, scs_log[layer], color=color, label=layer.replace('-',','))
        plt.title('logistic')
        plt.ylabel("score")
        plt.xlabel("alpha")
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(fontsize='xx-large')
        plt.subplot(1, 3, 3)
        plt.plot(ALPHA, scs_tan[layer], color=color, label=layer.replace('-',','))
        plt.title('tanh')
        plt.ylabel("score")
        plt.xlabel("alpha")
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(fontsize='xx-large')
    plt.show()
   