from PIL import Image
import os

# Fungsi untuk mengubah gambar menjadi warna hitam putih
def convert_to_grayscale(input_folder, output_folder):
    # Membuat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterasi melalui setiap kategori (moi dan rajaampat) di dalam train dan test
    for category in os.listdir(input_folder):
        category_path = os.path.join(input_folder, category)
        
        # Membuat folder output untuk kategori tertentu
        output_category_path = os.path.join(output_folder, category)
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        # Iterasi melalui setiap gambar di dalam kategori
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            
            # Membuka gambar menggunakan PIL
            img = Image.open(image_path)
            
            # Mengubah gambar menjadi skala abu-abu (grayscale)
            img = img.convert('L')
            
            # Menyimpan gambar hasil konversi
            output_path = os.path.join(output_category_path, image_name)
            img.save(output_path)

# Mengubah gambar-gambar dalam folder train menjadi warna hitam putih
# convert_to_grayscale('Dataset/Gallery', 'DatasetGrayscale/train_grayscale')

# Mengubah gambar-gambar dalam folder test menjadi warna hitam putih
# convert_to_grayscale('Dataset/Probe', 'DatasetGrayscale/test_grayscale')


# Fungsi untuk melakukan reshape pada gambar
def reshape_image(input_folder, output_folder, new_size):
    # Membuat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterasi melalui setiap kategori (moi dan rajaampat) di dalam train dan test
    for category in os.listdir(input_folder):
        category_path = os.path.join(input_folder, category)
        
        # Membuat folder output untuk kategori tertentu
        output_category_path = os.path.join(output_folder, category)
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        # Iterasi melalui setiap gambar di dalam kategori
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            
            # Membuka gambar menggunakan PIL
            img = Image.open(image_path)
            
            # Melakukan reshape pada gambar
            img_resized = img.resize(new_size)
            
            # Menyimpan gambar hasil reshape
            output_path = os.path.join(output_category_path, image_name)
            img_resized.save(output_path)

# Melakukan reshape pada gambar-gambar dalam folder train
# reshape_image('DatasetGrayscale/train_grayscale', 'DatasetGrayscale200/train_reshaped', (200, 200))

# Melakukan reshape pada gambar-gambar dalam folder test
# reshape_image('DatasetGrayscale/test_grayscale', 'DatasetGrayscale200/test_reshaped', (200, 200))

def reshape_image_single(input_folder, output_folder, new_size):
    # Membuat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterasi melalui setiap gambar di dalam folder input
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        
        # Membuka gambar menggunakan PIL
        img = Image.open(image_path)
        
        # Melakukan reshape pada gambar
        img_resized = img.resize(new_size)
        
        # Menyimpan gambar hasil reshape
        output_path = os.path.join(output_folder, image_name)
        img_resized.save(output_path)

reshape_image_single('face_images', 'face_image_reshaped', (240, 240))