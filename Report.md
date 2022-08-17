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

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.


