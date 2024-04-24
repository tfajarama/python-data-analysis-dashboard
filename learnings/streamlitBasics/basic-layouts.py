"""
Basic Layouts dalam Streamlit

Sejauh ini Anda telah mengenal berbagai hal mulai dari basic element hingga beberapa pilihan widget yang ada dalam
streamlit. Semua hal tersebut tentunya akan sangat membantu kita dalam membuat sebuah dashboard yang interaktif.

Apakah itu sudah cukup? Tentu saja belum!

Untuk membuat dashboard yang rapi, kita perlu belajar cara mengatur layout atau tampilan dari sebuah dashboard. Nah,
pada materi kali ini, kita akan mengupas tuntas tentang basic layout dalam streamlit.

Sebagai sebuah web app framework, streamlit telah menyediakan berbagai pilihan layout yang dapat digunakan untuk
mengatur tampilan web app (atau dashboard) yang akan dibuat. Pilihan layout yang tersedia antara lain sidebar, columns,
tabs, expander, serta container. Tentunya setiap pilihan layout tersebut memiliki fungsi dan kegunaannya masing-masing.
Yuk, kita bahas satu per satu!
"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

"""
Sidebar

Elemen layout pertama yang akan kita bahas ialah sidebar. Ia merupakan area yang berada di samping konten utama. Pada 
streamlit, posisi sidebar secara default berada di bagian kiri dari konten utama dan dapat memuat berbagai hal mulai 
dari gambar, teks, hingga widget.    

Untuk menambahkan sebuah elemen atau widget ke dalam sidebar, kita bisa menggunakan notasi with yang diikuti sebuah 
object bernama sidebar yang telah disediakan oleh streamlit. Berikut merupakan contoh kode untuk menambah sebuah elemen 
dan widget ke dalam sidebar.
"""
st.title('Belajar Analisis Data')

with st.sidebar:
    st.text('Ini merupakan sidebar')

    values = st.slider(
        label='Select a range of values',
        min_value=0, max_value=100, value=(0, 100)
    )
    st.write('Values:', values)

"""
Bagaimana cukup mudah bukan untuk membuat sidebar di streamlit? (dokumentasi: 
https://docs.streamlit.io/develop/api-reference/layout/st.sidebar). Nah, sidebar ini dapat kita gunakan untuk menampung 
gambar logo serta widget yang digunakan sebagai filter. Namun, hal ini akan kita bahas pada materi berikutnya.
"""

"""
Columns

Columns merupakan elemen layout yang memungkinkan kita untuk mengatur tampilan pada konten utama ke dalam beberapa kolom 
sesuai kebutuhan. Gambar berikut merupakan ilustrasi tampilan dari elemen layout ini. 

Untuk membuat column dalam streamlit app, kita harus membuat object dari setiap kolom terlebih dahulu. Hal ini dapat 
dilakukan dengan memanfaatkan function columns(). Selanjutnya, kita hanya perlu menambahkan sebuah elemen atau widget ke 
dalam column tersebut menggunakan notasi with. Berikut merupakan contoh kode untuk membuat column dalam streamlit.
"""
st.title('Belajar Analisis Data')
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Kolom 1")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
    st.header("Kolom 2")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    st.header("Kolom 3")
    st.image("https://static.streamlit.io/examples/owl.jpg")

"""
Sebenarnya, kita bisa dengan bebas mengatur ukuran dari column yang dibuat. Nah, untuk melakukan hal ini, kita harus 
memasukkan sebuah list yang berisi perbandingan ukuran dari kolom yang akan dibuat. Sebagai contoh, kode di bawah ini 
akan menampilkan 3 buah kolom  dengan perbandingan 2:1:1.
"""
st.title('Belajar Analisis Data')
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.header("Kolom 1")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
    st.header("Kolom 2")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    st.header("Kolom 3")
    st.image("https://static.streamlit.io/examples/owl.jpg")

"""
Tabs

Elemen layout berikutnya yang terdapat dalam streamlit ialah tabs. Ia merupakan elemen layout yang memungkinkan kita 
untuk menambahkan beberapa tabs ke dalam konten utama. Hal ini tentunya akan sangat membantu kita dalam menghasilkan 
dashboard yang interaktif.

Sama halnya dengan columns yang sebelumnya kita bahas, untuk membuat tabs, kita harus membuat object dari setiap tab 
menggunakan function tabs() yang diikuti dengan list dari nama dari setiap tab. Berikut contoh kode untuk membuat tabs 
dalam streamlit app.
"""
st.title('Belajar Analisis Data')
tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

with tab1:
    st.header("Tab 1")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with tab2:
    st.header("Tab 2")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with tab3:
    st.header("Tab 3")
    st.image("https://static.streamlit.io/examples/owl.jpg")

"""
Nah, itulah cara membuat tabs dalam streamlit app (dokumentasi: 
https://docs.streamlit.io/develop/api-reference/layout/st.tabs). Cukup mudah, bukan?
"""

"""
Container

Container merupakan elemen layout dalam streamlit yang memungkinkan kita untuk membuat sebuah wadah untuk menampung suatu 
atau beberapa elemen dengan ukuran yang tetap. Container ini dapat kita gunakan untuk menghasilkan dashboard yang rapi.

Untuk membuat container, kita bisa menggunakan notasi with dan diikuti function container(). Kode di bawah ini merupakan 
contoh kode untuk membuat container dalam streamlit app.
"""
with st.container():
    st.write("Inside the container")

    x = np.random.normal(15, 5, 250)

    fig, ax = plt.subplots()
    ax.hist(x=x, bins=15)
    st.pyplot(fig)

st.write("Outside the container")

"""
Ketika melihat gambar di atas, mungkin Anda tidak melihat adanya perbedaan dari tampilan sebelum dan sesudah menggunakan 
container. Hal ini terjadi karena kita masih belum memiliki banyak elemen yang ingin ditampilkan.
"""

"""
Expander

Elemen layout selanjutnya yang tidak kalah penting ialah expander. Anda dapat menganggap elemen layout ini sebagai sebuah 
container yang dapat diperluas atau diciutkan secara interaktif.

Untuk membuat elemen layout ini, kita bisa menggunakan notasi with dan diikuti dengan function expander() seperti pada 
contoh kode berikut.
"""
x = np.random.normal(15, 5, 250)

fig, ax = plt.subplots()
ax.hist(x=x, bins=15)
st.pyplot(fig)

with st.expander("See explanation"):
    st.write(
        """Lorem ipsum dolor sit amet, consectetur adipiscing elit, 
        sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris 
        nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor 
        in reprehenderit in voluptate velit esse cillum dolore eu fugiat 
        nulla pariatur. Excepteur sint occaecat cupidatat non proident, 
        sunt in culpa qui officia deserunt mollit anim id est laborum.
        """
    )

"""
Seperti yang terdapat pada gambar di atas, kita bisa menggunakan expander untuk menampung penjelasan atau keterangan 
lanjutan dari sebuah visualisasi data yang ditampilkan dalam sebuah dashboard.
"""

"""
Nah, itulah beberapa pilihan layout yang disediakan oleh streamlit untuk membantu kita dalam membuat web app yang rapi 
dan interaktif. Apabila ingin mempelajari topik ini lebih lanjut, Anda dapat mengunjungi dokumentasi berikut: 
https://docs.streamlit.io/develop/api-reference/layout.
"""
