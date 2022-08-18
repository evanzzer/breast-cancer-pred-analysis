# Laporan Proyek Machine Learning - Evans Hebert

## Domain Proyek

Kanker payudara merupakan kanker yang sering ditemukan dan terjadi dalam kalangan para wanita di seluruh dunia. Riset mengatakan bahwa sejumlah 25% dari seluruh kasus kanker merupakan kanker payudara dan jumlah kasus kanker yang mematikan ini sebanyak 2.1 juta orang hanya pada tahun 2015. 

Oleh karena itu, untuk melakukan identifikasi apakah seorang wanita yang menderita tumor pada bagian payudara adalah kanker atau tidak, tumor tersebut diidentifikasikan dan diklasifikasikan berdasarkan spesifikasi tertentu. Sebuah tumor dapat dikatakan sebagai *Malignant* atau kanker bila sel-sel tumor dapat tumbuh dan merambat ke sekitar tisu atau area yang dapat dijangkau dalam tubuh. Sebuah tumor dapat dikatakan *Benign* ketika tumor tidak menyerang tisu sekitar atau menyebar ke bagian tubuh lainnya seperti hal yang dilakukan oleh kanker. 

Klasifikasi dengan menggunakan Machine Learning ini sangat penting karena hal ini menyangkut apakah seseorang dapat bertahan hidup setelah mengidap tumor tersebut. Semakin cepat penanganannya, maka tingkat seseorang dapat bertahan hidup akan lebih tinggi. Setiap nyawa sangat berharga dan jika pengklasifikasian ini ternyata dapat membantu untuk mengkategorikannya, ditambah dengan penanganan yang tepat, maka dengan algoritma ini dapat menyelamatkan nyawa-nyawa.
  
Referensi:
- [Kaggle Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
- [ML Project: Breast Cancer Detection Using Machine Learning Classifier](https://indianaiproduction.com/breast-cancer-detection-using-machine-learning-classifier/)
- [Classification of Breast Cancer](https://muditkapoor01.medium.com/classification-of-breast-cancer-82d0058f3d35)

## Business Understanding

### Problem Statements

Agar dapat membuat sebuah penyelesaian dalam melakukan klasifikasi tumor yang merupakan *Malignant* atau *Benign*, maka hal-hal ini perlu diketahui dan diselesaikan. Hal-hal tersebut meliputi:
- Apa parameter yang paling mempengaruhi seseorang mengidap kanker payudara?
- Dari serangkaian algoritma yang ada, algoritma manakah yang paling akurat dalam memprediksikan bahwa seseorang mengidap kanker payudara?

### Goals

Tujuan yang perlu dicapai dalam pengklasifikasikan tumor adalah sebagai berikut:
- Mengetahui korelasi yang paling berdampak bahwa seseorang telah mengidap kanker payudara
- Mengetahui algoritma yang paling akurat dalam memprediksi bahwa seseorang telah mengidap kanker payudara.

### Solution statements
- Menggunakan teknik outliers dan correlations untuk mengidentifikasi ikatan yang paling berpengaruh dan menjadi fitur utama untuk melakukan vonis bahwa seseorang mengidap kanker payudara.
- Melakukan ujicoba antara 3 algoritma model, yaitu K-Nearest Neighbor, RandomForest, dan AdaBoost Algorithm.

## Data Understanding
Dataset yang akan digunakan berasal dari Kaggle yang berjudul [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset). Dataset tersebut terdiri dari 569 dataset dengan masing-masing data terdiri dari 32 kolom yang jika diuraikan meliputi hal-hal berikut ini.
- `id`: Unique ID
- `diagnosis`: M - Malignant (Cancerous), B - Benign (Non-cancerous)
- `radius_mean`: Radius of Lobes
- `texture_mean`: Mean of Surface texture
- `perimeter_mean`: Outer Perimeter of Lobes
- `area_mean`: Mean Area of Lobes
- `smoothness_mean`: Mean of Smoothness Levels
- `compactness_mean`: Mean of Compactness
- `concavity_mean`: Mean of Concavity
- `concave points_mean`: Mean of concave points
- `symmetry_mean`: Mean of Symmetry
- `fractal_dimension_mean`: Mean of Fractal Dimension
- `radius_se`: Standard Error of Radius
- `texture_se`: Standard Error of Texture
- `perimeter_se`: Standard Error of Perimeter
- `area_se`: Standard Error of Area
- `smoothness_se`: Standard Error of Smoothness
- `compactness_se`: Standard Error of Compactness
- `concavity_se`: Standard Error of Concavity
- `concave points_se`: Standard Error of Concave Points
- `symmetry_se`: Standard ErrorSE of Symmetry
- `fractal_dimension_se`: Standard Error of Fractal Dimension
- `radius_worst`: Worst Radius
- `texture_worst`: Worst Texture
- `perimeter_worst`: Worst Perimeter
- `area_worst`: Worst Area
- `smoothness_worst`: Worst Smoothness
- `compactness_worst`: Worst Compactness
- `concavity_worst`: Worst Concavity
- `concave points_worst`: Worst Concave Points
- `symmetry_worst`: Worst Symmetry
- `fractal_dimension_worst`: Worst Fractal Dimension

Dari kolom-kolom tersebut, seluruh fitur selain `id` yang diperoleh merupakan Numerical Features yang memiliki nilai berupa angka. Kolom `id` merupakan *unique identifier* yang tidak akan berguna dalam proses pengklasifikasian sehingga kolom tersebut dapat dihapus.

Pada dataset ini, terdapat kolom `diagnosis` yang merupakan output dari model Machine Learning yang akan menandakan seseorang mengidap penyakit kanker payudara atau tidak. Kolom `diagnosis` ini berbentuk object yang terdiri dari nilai `M` atau *Malignant* dan `B` atau *Benign*. Karena hanya terdiri dari dua kelas saja, maka kolom ini dapat dikonversi menjadi Boolean atau Integer dengan range nilai 0 dan 1, dengan 0 merepresentasikan `B` yang artinya seseorang belum mengidap kanker payudara, dan 1 merepresentasikan `M` yang artinya seseorang telah mengidap kanker payudara.

Pada proses ini, yang akan dikonsiderasikan merupakan fitur-fitur yang berhubungan dengan *mean* sehingga fitur-fitur yang berhubungan dengan *Standard Error* dan *Worst* dapat dihilangkan.

## Data Preparation
Pertama-tama, pengecekan nilai terhadap dataset perlu dilakukan untuk mendeteksi apakah terdapat data yang hilang atau bernilai `null`. Hal ini dapat dilihat dengan memanfaatkan fungsi `describe()` dari library `Pandas`. Jika terdapat nilai `null`, maka nilai minimal dari sebuah kolom/fitur bernilai 0. Alternatif lain untuk mendeteksi apakah terdapat nilai `null` adalah menggunakan fungsi `isnull().sum()` untuk menjumlahkan data yang bernilai null. Pada dataset ini, tidak ada nilai `null` sehingga dapat dilanjutkan ke proses selanjutnya.

Proses selanjutnya adalah mendeteksi outliers. Tujuan mendeteksi outliers adalah untuk menghilangkan data yang jauh sekali dari trend. Untuk melakukan analisis outliers, maka fungsi `boxplot()` dari library `seaborn` digunakan. Hasilnya dapat dilihat pada gambar di bawah ini.

![image](https://user-images.githubusercontent.com/56476347/185194462-cb443936-dcdf-4e08-9344-4ab6b592f547.png)

Dari gambar di atas, terdapat outliers yang sangat banyak, ditandai dengan gambar bulat di sebelah kanan. Untuk itu, perlu dilakukan pembersihan data dari outliers. Untuk itu, metode Inter Quartile Range digunakan untuk membersihkan data tersebut. Hasil pembersihan outliers menyisahkan 502 dataset yang dapat digunakan untuk melakukan pemodelan. Jika dijalankan fungsi `boxplot()` kembali, maka hasilnya adalah sebagai berikut.

![image](https://user-images.githubusercontent.com/56476347/185195006-82b9648f-334f-4807-a2ec-4981f0e1bc0f.png)

Masih terdapat outliers, namun sudah lebih sedikit daripada sebelumnya. Selanjutnya, dataset akan diuji korelasinya antar fitur dengan menggunakan fungsi `heatmap()` dari library `seaborn`. Korelasi ini penting dilakukan untuk menemukan seberapa erat hubungan fitur dengan label yang menjadi output dari model, yaitu `diagnosis`. Matrix kolerasi yang didapatkan dapat terlihat pada gambar di bawah ini.

![image](https://user-images.githubusercontent.com/56476347/185195305-64caab43-d37e-4e15-9c39-ebd2255ec26b.png)

Korelasi ini dapat terlihat bahwa:
- Fitur `texture_mean` memiliki korelasi yang paling kecil dengan fitur lainnya
- Fitur `radius_mean`, `perimeter_mean`, dan `area_mean` saling berkaitan erat satu dengan yang lainnya.
- Fitur `compactness_mean`, `concavity_mean`, dan `concave points_mean` saling berkaitan satu dengan lainnya.
- Fitur `concavity_mean` dan `concave points_mean` ada kaitannya dengan fitur `radius_mean`
- Fitur `fractal_dimension_mean` memiliki kaitan yang bersifat negatif dan lemah terhadap `diagnosis` sehingga fitur tersebut dapat dihilangkan dan tidak diikutsertakan dalam model.

Karena fitur `radius_mean`, `perimeter_mean`, dan `area_mean` saling berkaitan erat satu dengan yang lainnya, maka ketiga fitur tersebut dapat digabungkan menjadi satu fitur baru bernama `measurement_mean` dengan menggunakan Principal Component Analysis atau PCA. Hasil pengujian menyatakan bahwa fitur `radius_mean` dapat mendeskripsikan fitur `perimeter_mean` dan `area_mean` dengan nilai rasio 1:0:0. Hal ini karena secara rumus matematis, nilai `perimeter_mean` dan `area_mean` dihitung dari variabel `radius_mean`.

Setelah penggabungan fitur, maka dataset dapat dibagi menjadi train set dan test set dengan rasio antara keduanya adalah 8:2. Pembagian ini dapat dilakukan dengan mudah dengan menggunakan fungsi `train_test_split()` dari library `sklearn`. Hasil pembagiannya adalah train set terdiri dari 401 data dan test set terdiri dari 101 data. Selanjutnya, fitur-fitur tersebut dilakukan standarisasi untuk memudahkan model merumuskan algoritmanya yang akan digunakan. Cukup dengan memanggil class `StandardScaler()` dari library `sklearn` dapat mentransformasikan seluruh fitur sehingga setiap fitur memiliki rata-rata sebesar 0 dengan standar deviasi sebesar 1.

## Modeling
Algoritma pemodelan yang akan digunakan merupakan K-Nearest Neighbor, RandomForest, dan AdaBoost.

#### K-Nearest Neighbor
K-Nearest Neighbor merupakan sebuah algoritma yang mengukur jarak antara titk dengan titik lainnya dengan mengambil sejumlah titik terdekat yang telah ditentukan. Algoritma ini sederhana, namun akan menjadi masalah ketika fitur yang diuji sangat banyak. Parameter yang digunakan pada algoritma ini adalah `k = 10`, yaitu mengambil 10 titik terdekat. Metrik yang digunakan adalah metrik Minkowski dengan `p = 2`. Metrik ini sama dengan metrik Euclidean, dengan parameter `p` yang dapat diubah sesuai kebutuhan.

#### RandomForest
RandomForest merupakan sebuah algoritma yang merupakan model yang di dalamnya terdapat model-model yang bekerja secara bersamaan seperti model Decision Tree. Ibaratnya sebuah bagian yang kecil menyelesaikan pekerjaannya, dan bagian-bagian tersebut dikumpulkan hasilnya untuk memperoleh prediksi akhir. Parameter yang digunakan adalah `n_estimator = 20` yang artinya terdapat 20 pohon yang akan disediakan, dengan `max_depth = 16` yang artinya pembelahan maksimal dari sebuah pohon adalah 16 kali. Parameter `random_state = 55` mengatur *random number generator* yang digunakan, dan `n_jobs = -1` artinya seluruh proses akan berjalan secara paralel.

#### AdaBoost Algorithm
AdaBoost Algorithm merupakan model yang akan meningkatkan akurasi model secara iteratif. Algoritma ini akan terus dilatih hingga mencapai akurasi terbaik atau telah mencapai iterasi maksimum. Parameter-parameter yang digunakan meliputi `learning_rate = 0.05` yang merupakan bobot yang diterapkan untuk setiap regresi pada masing-masing iterasi yang dijalankan, serta `random_state = 55` untuk mengatur *random number generator* yang digunakan.

## Evaluation
Hasil pemodelan yang dijalankan dengan tiga algoritma tersebut dapat divisualisasi pada gambar di bawah ini.

![image](https://user-images.githubusercontent.com/56476347/185201163-a8329a98-fd77-42fa-aeeb-1fa8ea763e74.png)

Metrik yang digunakan merupakan pendekatan accuracy karena model ini melakukan prediksi binary classification apakah dengan fitur-fitur yang dispesifikasikan dapat dikategorikan menjadi kanker payudara atau tidak. Metrik akurasi didapatkan dengan menghitung jumlah data yang berhasil diprediksi dengan benar dan dibagi dengan jumlah total dari data yang tersedia. Karena hasil model memiliki output dari rentang 0 hingga 1 dalam format desimal, maka output tersebut akan dibulatkan menjadi nilai integer 0 dan 1, yaitu *Benign* atau *Malignant*

Hasil visualisasi ini diurutkan dari yang paling terkecil akurasinya hingga yang paling besar akurasi test set nya. Untuk algoritma KNN, didapatkan train accuracy sebesar 94.26% dan test accuracy sebesar 96.04%. Algoritma RandomForest didapatkan train accuracy sebesar 99% dan test accuracy sebesar 95.05%. Algoritma AdaBoost memiliki train accuracy sebesar 97.76% dan test accuracy sebesar 97.03%. 

Dari hasil tersebut, dapat disimpulkan bahwa algoritma AdaBoost merupakan model terbaik berdasarkan test set yang telah diuji. Ujicoba secara manual dengan mengambil sampling dari 10 data secara acak menyimpulkan bahwa model KNN dan AdaBoost memprediksi 9 dari 10 data secara benar, dan algoritma RandomForest memprediksikan 8 dari 10 data secara benar, sesuai dengan nilai akurasi yang telah didapatkan. 

**---Ini adalah bagian akhir laporan---**
