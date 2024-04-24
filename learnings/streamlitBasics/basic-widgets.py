"""
Basic Widgets dalam Streamlit

Yey, pada materi sebelumnya, kita telah mengenal berbagai basic element yang ada dalam streamlit. Hal tersebut tentunya
akan sangat membantu kita dalam membuat dashboard. Namun, untuk menghasilkan sebuah dashboard yang interaktif, kita
memerlukan komponen lain seperti widget (elemen Graphical User Interface yang memungkinkan pengguna untuk berinteraksi
dengan aplikasi).

Nah, pada materi ini, kita akan berkenalan dengan berbagai basic widget yang telah disediakan oleh streamlit. Untuk
membantu kita menghasilkan web app yang interaktif, streamlit telah menyediakan banyak sekali pilihan widget yang bisa
disesuaikan dengan kebutuhan. Agar lebih mudah dalam memahami seluruh widget tersebut, pada materi ini kita akan
membaginya ke dalam dua kategori, yaitu input widget dan button widget.
"""
import datetime
import pandas as pd
import streamlit as st

"""
Input Widget

Kategori widget yang akan kita bahas ialah input widget. Ia merupakan kategori widget yang memungkinkan pengguna untuk 
memberikan input ke dalam streamlit app. Terdapat beberapa widget yang termasuk kategori ini, seperti text input, number 
input, date input, dll. Yuk, kita bahas satu per satu!
"""

"""
Text input

Text input merupakan widget yang digunakan untuk memperoleh inputan berupa single-line text. Kita bisa menggunakan 
function text_input() untuk membuat widget ini. Berikut merupakan contoh kode untuk membuatnya.
"""
name = st.text_input(label='Nama lengkap', value='')
st.write('Nama: ', name)

"""
Text-area

Text area merupakan widget yang memungkinkan pengguna untuk menginput multi-line text. Untuk membuat widget ini, 
streamlit telah menyediakan function bernama text_area(). Kode di bawah ini merupakan contoh kode untuk menggunakan 
function tersebut.
"""
text = st.text_area('Feedback')
st.write('Feedback: ', text)

"""
Number input

Widget berikutnya yang akan kita bahas ialah number input . Ia merupakan widget yang digunakan untuk memperoleh inputan 
berupa angka dari pengguna. Anda dapat menggunakan contoh kode berikut untuk membuat number input widget menggunakan 
function number_input().
"""
number = st.number_input(label='Umur')
st.write('Umur: ', int(number), ' tahun')

"""
Date input

Selain inputan berupa angka dan teks, pada beberapa kasus kita juga membutuhkan input berupa tanggal dari pengguna 
melalui date input widget. Kita dapat membuat widget tersebut menggunakan function date_input(). Berikut merupakan 
contoh kode untuk menggunakannya.
"""
date = st.date_input(label='Tanggal lahir', min_value=datetime.date(1900, 1, 1))
st.write('Tanggal lahir:', date)

"""
File uploader

Widget selanjutnya yang akan kita bahas ialah file uploader. Widget ini memungkinkan kita meminta pengguna untuk 
meng-upload sebuah berkas tertentu ke dalam web app. Kita dapat membuat file uploader widget menggunakan function 
file_uploader() seperti pada contoh kode berikut. 
"""
uploaded_file = st.file_uploader('Choose a CSV file')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

"""
Camera input

Selain file uploader, streamlit juga menyediakan camera input widget yang dapat digunakan untuk meminta user mengambil 
gambar melalui webcam sekaligus mengunggahnya.  Hal ini tentunya dilakukan dengan persetujuan pengguna. Berikut 
merupakan contoh kode untuk membuat camera input widget menggunakan function camera_input().
"""
picture = st.camera_input('Take a picture')
if picture:
    st.image(picture)

"""
Button Widgets
 
Oke, kategori widget selanjutnya yang akan kita bahas ialah button widget. Ia merupakan kategori widget yang terdiri 
dari button, checkbox, radio button, dll.
"""

"""
Button

Button merupakan widget untuk menampilkan tombol interaktif. Tombol tersebut dapat digunakan pengguna untuk melakukan 
aksi tertentu. Untuk membuat widget ini, kita bisa menggunakan function button() seperti contoh berikut.
"""
if st.button('Say hello'):
    st.write('Hello there')

"""
Checkbox
Checkbox merupakan widget yang digunakan untuk menampilkan sebuah checklist untuk pengguna. Kita bisa menggunakan 
function checkbox() untuk membuat dan menampilkan checklist dalam streamlit app. Berikut contoh kode untuk melakukannya.
"""
agree = st.checkbox('I agree')

if agree:
    st.write('Welcome to MyApp')

"""
Radio button

Selain button dan checkbox, terkadang kita juga membutuhkan radio button untuk menghasilkan web app yang interaktif. Ia 
merupakan widget yang memungkinkan pengguna untuk memilih satu dari beberapa pilihan yang ada. Untuk membuat widget ini, 
kita bisa menggunakan function radio() seperti pada contoh kode berikut.
"""
genreRadio = st.radio(
    label="What's your favorite movie genre",
    options=('Comedy', 'Drama', 'Documentary'),
    horizontal=False
)

"""
Select Box

Select box merupakan widget yang memungkinkan pengguna untuk memilih salah satu dari beberapa pilihan yang ada. Ia 
merupakan opsi alternatif dari radio button. Streamlit telah menyediakan function selectbox() untuk membuat select box 
widget. Berikut contoh penggunaannya.
"""
genreSelect = st.selectbox(
    label="What's your favorite movie genre",
    options=('Comedy', 'Drama', 'Documentary')
)

"""
Multiselect

Widget lain yang harus kita ketahui ialah multiselect. Ia merupakan widget yang digunakan agar user dapat memilih lebih 
dari satu pilihan dari sekumpulan pilihan yang ada. Untuk mempermudah kita dalam membuat multiselect widget, streamlit 
telah menyediakan function bernama multiselect(). Berikut contoh kode untuk menggunakan function tersebut.
"""
genreMultiselect = st.multiselect(
    label="What's your favorite movie genre",
    options=('Comedy', 'Drama', 'Documentary')
)

"""
Slider

Slider merupakan widget yang memungkinkan pengguna untuk untuk memilih nilai (atau range nilai) dari sebuah range nilai 
yang telah ditentukan. Streamlit telah menyediakan function slider() untuk membuat slider widget. Berikut merupakan 
contoh penggunaannya.
"""
values = st.slider(
    label='Select a range of values',
    min_value=0, max_value=100, value=(0, 100))
st.write('Values:', values)

"""
Nah, itulah beberapa pilihan widget yang disediakan oleh streamlit. Anda dapat melihat lebih banyak pilihan widget pada 
dokumentasi berikut: https://docs.streamlit.io/develop/api-reference/widgets
"""
