

from store_vectors import get_extract_model, extract_vector
import pickle
from matplotlib import pyplot as plt
import faiss
import sys

# Thiết lập encoding utf-8 cho stdout
sys.stdout.reconfigure(encoding='utf-8')


loaded_index = faiss.read_index("training-faiss.index")

path_images = pickle.load(open('paths.pkl', 'rb'))

image_path_search = 'dataset/2.jpg'


model = get_extract_model()

vector_search = extract_vector(model, image_path_search)
vector_search = vector_search.reshape(1, -1)


k = 5

D, I = loaded_index.search(vector_search, k)

print("Kết quả tìm kiếm:")
print("Khoảng cách:", D)
print("Chỉ số hàng xóm:", I)

for neighbor_idx, distance in zip(I[0], D[0]):
    # Đường dẫn tới hình ảnh hàng xóm
    neighbor_image_path = path_images[neighbor_idx]
    neighbor_image = plt.imread(neighbor_image_path)  # Đọc hình ảnh

    # Hiển thị hình ảnh hàng xóm
    plt.figure()
    plt.imshow(neighbor_image)
    plt.title(f"Hinh anh hang xom - Khoang cach: {distance}")
    plt.show()
