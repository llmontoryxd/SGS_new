import os
from view import App

if not os.path.exists('outdata'):
    os.mkdir('outdata')
app = App()
app.mainloop()
