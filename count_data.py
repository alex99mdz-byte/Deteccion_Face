import os

def count_files(base):
    if not os.path.isdir(base):
        print(base, 'NO EXISTE')
        return
    for d in sorted(os.listdir(base)):
        path = os.path.join(base, d)
        if os.path.isdir(path):
            cnt = sum(1 for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
            print(base, d, cnt)

count_files('datasets/edades')
print('---')
count_files('datasets/emociones_procesado')
