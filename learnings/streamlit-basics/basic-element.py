"""
Basic Element dalam Streamlit

Sebelumnya, Anda telah berhasil menginstal dan membuat web app sederhana menggunakan streamlit. Untuk melengkapi
pengetahuan Anda tentang streamlit, kita akan mengenal berbagai basic element yang terdapat dalam streamlit pada
materi kali ini.

Sebagai sebuah web app framework yang andal, streamlit telah menyediakan banyak pilihan element, widget, layout serta
container untuk memastikan kita dapat membuat web app atau dashboard yang menarik dan interaktif. Nah, pada materi
ini, kita hanya akan fokus pada bagian basic element dalam streamlit yang terdiri dari write, text, data display, dan
chat. Yuk, kita bahas satu per satu!
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

"""
Write

Basic element pertama yang akan kita bahas ialah write. Ia merupakan elemen streamlit yang digunakan untuk menampilkan 
sebuah output.  Untuk menggunakan element ini, kita hanya perlu memanggil function write() dan diikuti inputan yang 
ingin ditampilkan.

Pada contoh pembuatan “Hello, world!” app, kita telah menggunakan function write() untuk menampilkan output dari 
argument markdown.  Sebenarnya function ini dapat digunakan untuk menampilkan hal lain, seperti DataFrame, visualisasi 
data, dll. Berikut merupakan contoh penggunaannya untuk menampilkan DataFrame.
"""
st.write(pd.DataFrame({
    'c1': [1, 2, 3, 4],
    'c2': [10, 20, 30, 40],
}))

"""
Text

Elemen lain yang ada dalam streamlit ialah text (dokumentasi: text). Sesuai dengan namanya, ia merupakan elemen yang 
digunakan untuk menampilkan sebuah output berupa text. Elemen ini memiliki banyak function yang bisa digunakan sesuai 
kebutuhan. Berikut merupakan beberapa pilihan yang tersedia saat ini.
"""

"""
markdown()

Function ini digunakan untuk menampilkan output dari argument markdown. Berikut merupakan contoh kode untuk 
menggunakannya.
"""
st.markdown(
    """
    # My first app
    Hello, para calon praktisi data masa depan!
    """
)

"""
title()

Ini merupakan function yang digunakan untuk menampilkan teks dalam format judul. Kode yang dapat Anda gunakan untuk 
menerapkan function tersebut adalah seperti di bawah ini.
"""
st.title('Belajar Analisis Data')

"""
header()

Function ini digunakan untuk menampilkan output teks sebagai format header. Berikut contoh penulisan kode untuk 
menggunakannya.
"""
st.header('Pengembangan Dashboard')

"""
subheader()

Ini merupakan function yang digunakan untuk menampilkan output teks sebagai format subheader.
"""
st.subheader('Pengembangan Dashboard')

"""
caption()

Function berikutnya ialah caption(). Ia merupakan function yang digunakan untuk menampilkan output teks dalam ukuran 
kecil. Function ini biasanya digunakan untuk menampilkan caption, footnotes, dll. Contoh penggunaannya seperti di bawah 
ini.
"""
st.caption('Copyright (c) 2023')

"""
code()

Pada beberapa case, kita harus menampilkan potongan kode ke dalam streamlit app (web app yang dibuat menggunakan 
streamlit). Untuk menjawab hal ini, streamlit telah menyediakan sebuah function bernama code(). Kode di bawah ini 
merupakan contoh penggunaan dari function tersebut.
"""
code = """def hello():
    print("Hello, Streamlit!")"""
st.code(code, language='python')

"""
text()

Function selanjutnya ialah text(). Function ini digunakan untuk menampilkan sebuah normal teks. Berikut merupakan contoh 
kode untuk mengguankannya.
"""
st.text('Halo, calon praktisi data masa depan.')

"""
latex()

Function terakhir yang dapat digunakan untuk menampilkan elemen teks ialah latex(). Sesuai namanya, function tersebut 
digunakan untuk menampilkan mathematical expression yang ditulis dalam format LaTeX. Berikut contoh kode untuk 
menggunakan function latex().
"""
st.latex(r"""
    \sum_{k=0}^{n-1} ar^k =
    a \left(\frac{1-r^{n}}{1-r}\right)
""")

"""
Data Display

Basic element selanjutnya yang akan kita bahas ialah data display (dokumentasi: data display). Ia merupakan elemen yang 
digunakan untuk menampilkan data secara cepat dan interaktif ke dalam streamlit app yang kita buat. Elemen ini memiliki 
beberapa function seperti berikut.
"""

"""
dataframe()

Function pertama yang bisa kita gunakan untuk menampilkan data ke dalam streamlit app ialah dataframe(). Ia merupakan 
function yang digunakan untuk menampilkan DataFrame sebagai sebuah tabel interaktif. Pada function ini, kita bisa 
mengatur ukuran dari table yang ingin ditampilkan menggunakan parameter width dan height. Berikut merupakan contoh kode 
untuk menampilkan data menggunakan function dataframe().
"""
df = pd.DataFrame({
    'c1': [1, 2, 3, 4],
    'c2': [10, 20, 30, 40],
})

st.dataframe(data=df, width=500, height=150)

"""
table()

Selain dataframe(), kita juga bisa menggunakan function table()untuk menampilkan data ke dalam streamlit app. Ia dapat 
digunakan untuk menampilkan data dalam bentuk static table. Berikut merupakan contoh penggunaannya.
"""
df = pd.DataFrame({
    'c1': [1, 2, 3, 4],
    'c2': [10, 20, 30, 40],
})
st.table(data=df)

"""
metric()

Ketika membuat dashboard, terkadang kita perlu menampilkan sebuah metrik tertentu. Untuk melakukan hal ini, kita bisa 
memanfaatkan function metric(). Function ini dapat membantu kita untuk menampilkan sebuah metrik tertentu beserta 
detailnya seperti label, value serta besar perubahan nilainya. Berikut merupakan contoh kode untuk menggunakannya.
"""
st.metric(label="Temperature", value="28 °C", delta="1.2 °C")

"""
json()

Selain bentuk DataFrame atau tabel, terkadang kita juga perlu menampilkan data dalam format JSON. Streamlit telah 
menyediakan function json() untuk menampilkan data dalam format JSON. Berikut merupakan contoh penggunaannya.
"""
st.json({
    'c1': [1, 2, 3, 4],
    'c2': [10, 20, 30, 40],
})

"""
Chart

Basic element terakhir yang perlu kita ketahui ialah chart. Sesuai namanya, elemen ini dapat digunakan untuk menampilkan 
grafik visualisasi data ke dalam streamlit app. Elemen inilah yang akan sering kita gunakan untuk membuat dashboard 
nantinya. Sebenarnya streamlit telah menyediakan banyak sekali function untuk mendukung berbagai library visualisasi 
data (dokumentasi: chart). Namun, pada materi ini, kita hanya akan fokus pada function pyplot().

Function pyplot() dapat digunakan untuk menampilkan grafik visualisasi data yang dibuat menggunakan matplotlib. Berikut 
merupakan contoh kode untuk menggunakannya.
"""
x = np.random.normal(15, 5, 250)

fig, ax = plt.subplots()
ax.hist(x=x, bins=15)
st.pyplot(fig)
