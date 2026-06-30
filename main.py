import tkinter as tk
from tkinter import messagebox, colorchooser
from tkinter import ttk
import cv2
import os
import numpy as np
import time
import json
import re
import webbrowser
from tkcalendar import DateEntry  # Librería para los calendarios

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "cascades",
    "haarcascade_frontalface_default.xml"
)

if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(
        f"No se encontró el clasificador Haar:\n{CASCADE_PATH}"
    )

# --- PALETA DE COLORES ---
COLOR_DEEP_BG = "#16060a"       # Fondo oscuro profundo
COLOR_SURFACE_CARD = "#2d3748"  # Fondo de la tarjeta central
COLOR_PRIMARY_RED = "#f41515"   # Rojo carmesí principal
COLOR_TEXT_MAIN = "#ffffff"     # Texto blanco principal
COLOR_TEXT_SEC = "#a0aec0"      # Texto secundario de apoyo
COLOR_INPUT_BG = "#364357"       # Fondo interno para los inputs
COLOR_BOX_BG = "#1e242f"         # Fondo oscuro para la caja de hobbies

# Configuración del archivo de Base de Datos JSON
USERS_JSON = "usuarios.json"
if not os.path.exists(USERS_JSON):
    with open(USERS_JSON, "w") as f:
        json.dump({}, f)

USERS_DIR = "users_lbph"
if not os.path.exists(USERS_DIR):
    os.makedirs(USERS_DIR)

# --- VARIABLES DE ESTADO ---
failed_attempts = 0
lockout_time = 0  # Tiempo en el que se desbloqueará el sistema
# Variables globales para persistencia de configuración
config_dificultad = "Medio"
config_tiempo = "1.5 minutos"

# --- FUNCIÓN AUXILIAR OJO ---
def toggle_password(entry_widget):
    if entry_widget.cget("show") == "":
        entry_widget.config(show="*")
    else:
        entry_widget.config(show="")

# --- GESTIÓN DE BASE DE DATOS (JSON) ---

def cargar_usuarios():
    try:
        with open(USERS_JSON, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def guardar_usuarios_db(data):
    with open(USERS_JSON, "w") as f:
        json.dump(data, f, indent=4)

def guardar_nuevo_usuario(username, password, nombre, fecha_nac, genero, hobbies, pregunta, respuesta, tarjeta, vencimiento_tarjeta):
    usuarios = cargar_usuarios()
    if username in usuarios:
        return False
    
    # Censura para almacenamiento: "**** **** **** 1234"
    tarjeta_censurada = "**** **** **** " + tarjeta.replace("-", "").replace(" ", "")[-4:]
    
    usuarios[username] = {
        "password": password,
        "nombre": nombre,
        "fecha_nacimiento": fecha_nac,
        "genero": genero,
        "hobbies": hobbies,
        "pregunta_seguridad": pregunta,
        "respuesta_seguridad": respuesta,
        "tarjeta_membresia": tarjeta_censurada,
        "vencimiento_tarjeta": vencimiento_tarjeta
    }
    guardar_usuarios_db(usuarios)
    return True


# --- RECONOCIMIENTO FACIAL ---

def cargar_rostros_conocidos():
    encodings = []
    nombres = []
    for archivo in os.listdir(USERS_DIR):
        if archivo.endswith(".npy"):
            nombre_usuario = os.path.splitext(archivo)[0]
            if not nombre_usuario.strip():
                continue
            ruta = os.path.join(USERS_DIR, archivo)
            vector = np.load(ruta).flatten()
            encodings.append(vector)
            nombres.append(nombre_usuario)
    return encodings, nombres


def registrar_rostro(username):
    if not username.strip():
        messagebox.showwarning("Falta información", "Escribe un nombre de usuario primero para asociar tu rostro.")
        return False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cámara no disponible.")
        return False
        
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    muestras = []   
    contador = 0

    messagebox.showinfo(
        "Escaneo Facial 2D",
        "Se abrirá la cámara de juego. Mira fijamente al lente.\n"
        "Se tomarán 10 muestras automáticamente."
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "No se pudo acceder a la cámara.")
            cap.release()
            cv2.destroyAllWindows()
            return False

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = cascade.detectMultiScale(gris, 1.3, 5)

        for (x, y, w, h) in caras:
            cara = cv2.resize(gris[y:y+h, x:x+w], (100, 100))
            muestras.append(cara)
            contador += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (244, 21, 21), 2)
            cv2.putText(frame, f"Captura {contador}/10", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if contador == 1:
            cv2.namedWindow("Analizando Rostro...", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Analizando Rostro...", 400, 200)

        cv2.imshow("Analizando Rostro...", frame)

        if contador >= 10 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if muestras:
        vector_promedio = np.mean(muestras, axis=0)
        ruta = os.path.join(USERS_DIR, f"{username}.npy")
        np.save(ruta, vector_promedio)
        messagebox.showinfo("Éxito", f"¡Rostro de '{username}' vinculado perfectamente!")
        return True
    else:
        messagebox.showwarning("Escaneo Fallido", "No se detectó el rostro de batalla. Intenta de nuevo.")
        return False


def login_con_rostro(callback_exito):
    global failed_attempts, lockout_time
    
    # Verificar bloqueo
    if time.time() < lockout_time:
        restante = int(lockout_time - time.time())
        messagebox.showerror("SISTEMA BLOQUEADO", f"Sistema bloqueado. Intenta de nuevo en {restante} segundos.")
        return

    encodings, nombres = cargar_rostros_conocidos()
    if not encodings:
        messagebox.showerror("Error", "No hay registros faciales en la base de datos.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cámara no disponible.")
        return

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    inicio = time.time()
    
    cv2.namedWindow("Escaneo de Acceso", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Escaneo de Acceso", 400, 200)

    def actualizar_frame():
        global failed_attempts, lockout_time
        ret, frame = cap.read()
        if not ret:
            finalizar()
            return

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = cascade.detectMultiScale(gris, 1.3, 5)
        username_reconocido = None

        for (x, y, w, h) in caras:
            cara_plana = cv2.resize(gris[y:y+h, x:x+w], (100, 100)).flatten()
            distancias = [np.linalg.norm(cara_plana - enc) for enc in encodings]
            
            if distancias:
                min_distancia = min(distancias)
                mejor_indice = int(np.argmin(distancias))

                if min_distancia < 3000:
                    username_reconocido = nombres[mejor_indice]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Acceso Concedido: {username_reconocido}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 244), 2)
                    cv2.putText(frame, "Desconocido", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 244), 2)

        cv2.imshow("Escaneo de Acceso", frame)

        if username_reconocido:
            failed_attempts = 0
            ventana.after(1000, lambda: finalizar_con_exito(username_reconocido))
            return

        if cv2.waitKey(1) & 0xFF == ord('q'):
            finalizar()
            return

        if time.time() - inicio > 10:
            finalizar()
            failed_attempts += 1
            if failed_attempts >= 3:
                lockout_time = time.time() + 180
                messagebox.showerror("SISTEMA BLOQUEADO", "3 intentos fallidos. Sistema bloqueado por 3 minutos.")
            else:
                messagebox.showinfo("Tiempo Excedido", "No se pudo reconocer tu rostro, 10 segundos excedidos.")
            return

        ventana.after(10, actualizar_frame)

    def finalizar():
        cap.release()
        cv2.destroyAllWindows()

    def finalizar_con_exito(user):
        finalizar()
        callback_exito(user)

    actualizar_frame()


# --- INTERFAZ GRÁFICA INTERACTIVA ---

def intentar_login(entry_usuario, entry_password):
    global failed_attempts, lockout_time
    
    # Verificar bloqueo
    if time.time() < lockout_time:
        restante = int(lockout_time - time.time())
        messagebox.showerror("SISTEMA BLOQUEADO", f"Sistema bloqueado. Intenta de nuevo en {restante} segundos.")
        return

    usuario = entry_usuario.get().strip()
    password = entry_password.get()

    if not usuario:
        return

    usuarios = cargar_usuarios()
    if usuario in usuarios and usuarios[usuario]["password"] == password:
        messagebox.showinfo("¡VICTORIA!", f"¡Bienvenido de vuelta a la arena, {usuarios[usuario]['nombre']}!")
        failed_attempts = 0
    else:
        failed_attempts += 1
        if failed_attempts >= 3:
            lockout_time = time.time() + 180
            messagebox.showerror("SISTEMA BLOQUEADO", "SISTEMA BLOQUEADO: Has excedido los 3 intentos permitidos. Sistema bloqueado por 3 minutos.")
        else:
            messagebox.showerror("Error de Acceso", f"Credenciales incorrectas. Intento {failed_attempts}/3.")


def al_reconocer(username):
    usuarios = cargar_usuarios()
    if username in usuarios:
        messagebox.showinfo("¡ACCESO CONCEDIDO!", f"¡Bienvenido por reconocimiento facial, {usuarios[username]['nombre']}!")
    else:
        messagebox.showerror("Error", "Rostro reconocido pero sin perfil activo.")


def recover_password():
    usuarios = cargar_usuarios()
    
    ventana_rec = tk.Toplevel(ventana)
    ventana_rec.title("Recuperación de Contraseña")
    ventana_rec.configure(bg=COLOR_SURFACE_CARD)
    
    ventana_rec.resizable(True, True)
    maximizar_ventana(ventana_rec)
    
    # Botón X
    btn_x = tk.Button(ventana_rec, text="X", font=("Arial", 12, "bold"), bg=COLOR_PRIMARY_RED, fg="white", bd=0, command=ventana_rec.destroy)
    btn_x.place(relx=1.0, rely=0.0, x=-10, y=10, anchor="ne")
    
    card_rec = tk.Frame(ventana_rec, bg=COLOR_SURFACE_CARD, bd=1, relief="solid")
    card_rec.place(relx=0.5, rely=0.5, anchor="center", width=400, height=300)
    
    tk.Label(card_rec, text="RECUPERACIÓN DE CUENTA", font=("Segoe UI", 12, "bold"), fg=COLOR_PRIMARY_RED, bg=COLOR_SURFACE_CARD).pack(pady=15)
    tk.Label(card_rec, text="Ingresa tu Nombre de Usuario:", font=("Segoe UI", 9), fg=COLOR_TEXT_MAIN, bg=COLOR_SURFACE_CARD).pack()
    
    ent_user = tk.Entry(card_rec, font=("Segoe UI", 10), bg=COLOR_INPUT_BG, fg=COLOR_TEXT_MAIN, bd=0, highlightthickness=1, highlightbackground=COLOR_TEXT_SEC, highlightcolor=COLOR_PRIMARY_RED)
    ent_user.pack(pady=5, ipady=4, width=250)
    
    def paso_2():
        user = ent_user.get().strip()
        if user not in usuarios:
            messagebox.showerror("Error", "El usuario no existe.")
            return
        
        pregunta = usuarios[user]["pregunta_seguridad"]
        resp_correcta = usuarios[user]["respuesta_seguridad"]
        
        for w in card_rec.winfo_children(): w.destroy()
        
        tk.Label(card_rec, text="PREGUNTA DE SEGURIDAD", font=("Segoe UI", 11, "bold"), fg=COLOR_PRIMARY_RED, bg=COLOR_SURFACE_CARD).pack(pady=15)
        tk.Label(card_rec, text=pregunta, font=("Segoe UI", 10, "italic"), fg=COLOR_TEXT_MAIN, bg=COLOR_SURFACE_CARD, wraplength=350).pack(pady=5)
        
        ent_resp = tk.Entry(card_rec, font=("Segoe UI", 10), bg=COLOR_INPUT_BG, fg=COLOR_TEXT_MAIN, bd=0, highlightthickness=1, highlightbackground=COLOR_TEXT_SEC, highlightcolor=COLOR_PRIMARY_RED)
        ent_resp.pack(pady=5, ipady=4, width=250)
        
        def verificar_respuesta():
            if ent_resp.get().strip().lower() == resp_correcta.lower():
                messagebox.showinfo("Validado", f"¡Validación correcta!\nTu contraseña es: {usuarios[user]['password']}")
                ventana_rec.destroy()
            else:
                messagebox.showerror("Error", "Respuesta incorrecta de seguridad.")
        
        btn_v = tk.Button(card_rec, text="VERIFICAR RESPUESTA", font=("Segoe UI", 9, "bold"), bg=COLOR_PRIMARY_RED, fg=COLOR_TEXT_MAIN, bd=0, command=verificar_respuesta)
        btn_v.pack(pady=15, ipady=5, width=200)

    tk.Button(card_rec, text="BUSCAR PREGUNTA", font=("Segoe UI", 9, "bold"), bg=COLOR_PRIMARY_RED, fg=COLOR_TEXT_MAIN, bd=0, command=paso_2).pack(pady=15, ipady=5, width=180)


def configuring_efecto_hover(boton, color_normal, color_hover):
    boton.bind("<Enter>", lambda e: boton.config(bg=color_hover))
    boton.bind("<Leave>", lambda e: boton.config(bg=color_normal))


def maximizar_ventana(win):
    win.update()
    try:
        win.attributes('-zoomed', True)
    except tk.TclError:
        try:
            win.state('zoomed')
        except tk.TclError:
            pass


# --- VISTA: PERSONALIZACIÓN ---

def mostrar_personalizacion():
    global vista_actual
    vista_actual = "personalizacion"
    for widget in contenedor_principal.winfo_children():
        widget.destroy()
    
    # Botón X
    btn_x = tk.Button(contenedor_principal, text="X", font=("Arial", 10), bg=COLOR_PRIMARY_RED, fg="white", bd=0, command=mostrar_lobby)
    btn_x.place(relx=1.0, rely=0.0, x=-10, y=10, anchor="ne")
    
    btn_settings.config(command=mostrar_lobby)
    contenedor_principal.place(relx=0.5, rely=0.5, anchor="center", width=450, height=450)

    tk.Label(contenedor_principal, text="PERSONALIZACIÓN", font=("Segoe UI", 16, "bold"), fg=COLOR_PRIMARY_RED, bg=COLOR_SURFACE_CARD).pack(pady=(30, 20))
    tk.Label(contenedor_principal, text="1. Elige tu Color Base", font=("Segoe UI", 9), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(anchor="w", padx=40)

    color_actual = tk.StringVar(value=COLOR_PRIMARY_RED)

    def cambiar_color():
        c = colorchooser.askcolor(title="Selecciona color base")[1]
        if c:
            color_actual.set(c)
            btn_sphere.config(highlightbackground=c)

    btn_sphere = tk.Button(contenedor_principal, text="🎨", font=("Segoe UI", 24), bg=COLOR_INPUT_BG, fg=COLOR_TEXT_MAIN, 
                           highlightbackground=COLOR_PRIMARY_RED, highlightthickness=4, bd=0, cursor="hand2", 
                           command=cambiar_color, width=6, height=3)
    btn_sphere.pack(pady=20)

    btn_apply = tk.Button(contenedor_principal, text="APLICAR CAMBIOS", font=("Segoe UI", 10, "bold"), bg=COLOR_PRIMARY_RED, fg=COLOR_TEXT_MAIN, bd=0, command=lambda: messagebox.showinfo("Éxito", f"Paleta actualizada a {color_actual.get()}"))
    btn_apply.pack(fill="x", padx=40, ipady=10, pady=20)
    configuring_efecto_hover(btn_apply, COLOR_PRIMARY_RED, "#ff3333")


# --- VISTA: LOBBY PRINCIPAL ---

def mostrar_lobby():
    global vista_actual
    vista_actual = "lobby"
    ventana.minsize(400, 500)
    for widget in contenedor_principal.winfo_children():
        widget.destroy()
        
    contenedor_principal.place(relx=0.5, rely=0.5, anchor="center", width=500, height=430)
    btn_settings.config(command=abrir_modal_configuracion)

    tk.Label(contenedor_principal, text="AVATARS VS ROOKS", font=("Segoe UI", 20, "bold"), fg=COLOR_PRIMARY_RED, bg=COLOR_SURFACE_CARD).pack(pady=(40, 2))
    tk.Label(contenedor_principal, text="BIENVENIDO, USUARIO", font=("Segoe UI", 11, "bold"), fg=COLOR_TEXT_MAIN, bg=COLOR_SURFACE_CARD).pack(pady=(0, 40))

    btn_pers = tk.Button(contenedor_principal, text="PERSONALIZACIÓN", font=("Segoe UI", 11, "bold"), bg=COLOR_PRIMARY_RED, fg=COLOR_TEXT_MAIN, bd=0, cursor="hand2", command=mostrar_personalizacion)
    btn_pers.pack(fill="x", padx=35, ipady=10, pady=8)
    configuring_efecto_hover(btn_pers, COLOR_PRIMARY_RED, "#ff3333")

    btn_offline = tk.Button(contenedor_principal, text="JUEGO OFFLINE", font=("Segoe UI", 11, "bold"), bg=COLOR_PRIMARY_RED, fg=COLOR_TEXT_MAIN, bd=0, cursor="hand2", command=lambda: messagebox.showinfo("Offline", f"Iniciando Juego Offline...\nDificultad: {config_dificultad} | Tiempo: {config_tiempo}"))
    btn_offline.pack(fill="x", padx=35, ipady=10, pady=8)
    configuring_efecto_hover(btn_offline, COLOR_PRIMARY_RED, "#ff3333")

    btn_online = tk.Button(contenedor_principal, text="JUEGO ONLINE", font=("Segoe UI", 11, "bold"), bg=COLOR_PRIMARY_RED, fg=COLOR_TEXT_MAIN, bd=0, cursor="hand2", command=mostrar_login)
    btn_online.pack(fill="x", padx=35, ipady=10, pady=8)
    configuring_efecto_hover(btn_online, COLOR_PRIMARY_RED, "#ff3333")
    
    maximizar_ventana(ventana)


def abrir_modal_configuracion():
    global config_dificultad, config_tiempo
    modal = tk.Toplevel(ventana)
    modal.title("Configuración")
    modal.geometry("340x250")
    modal.configure(bg=COLOR_SURFACE_CARD)
    modal.resizable(False, False)
    modal.transient(ventana)
    modal.grab_set()
    
    # Botón X
    btn_x = tk.Button(modal, text="X", font=("Arial", 10), bg=COLOR_PRIMARY_RED, fg="white", bd=0, command=modal.destroy)
    btn_x.place(relx=1.0, rely=0.0, x=-10, y=10, anchor="ne")
    
    x = ventana.winfo_x() + (ventana.winfo_width() // 2) - 170
    y = ventana.winfo_y() + (ventana.winfo_height() // 2) - 150
    modal.geometry(f"+{x}+{y}")

    tk.Label(modal, text="CONFIGURACIÓN", font=("Segoe UI", 12, "bold"), fg=COLOR_PRIMARY_RED, bg=COLOR_SURFACE_CARD).pack(pady=(15, 10))
    tk.Label(modal, text="Nivel de Dificultad", font=("Segoe UI", 9), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(anchor="w", padx=30, pady=(5, 2))
    
    def actualizar_config(event):
        global config_dificultad, config_tiempo
        config_dificultad = combo_dif.get()
        if config_dificultad == "Fácil":
            config_tiempo = "1 minuto"
        elif config_dificultad == "Medio":
            config_tiempo = "1.5 minutos"
        elif config_dificultad == "Difícil":
            config_tiempo = "2 minutos"

    combo_dif = ttk.Combobox(modal, values=["Fácil", "Medio", "Difícil"], state="readonly", font=("Segoe UI", 10))
    combo_dif.set(config_dificultad)
    combo_dif.bind("<<ComboboxSelected>>", actualizar_config)
    combo_dif.pack(fill="x", padx=30)

    btn_cerrar = tk.Button(modal, text="GUARDAR Y CERRAR", font=("Segoe UI", 9, "bold"), bg=COLOR_PRIMARY_RED, fg=COLOR_TEXT_MAIN, bd=0, cursor="hand2", command=modal.destroy)
    btn_cerrar.pack(pady=25, ipady=6, width=160)


# --- VISTA: INICIO DE SESIÓN ---

def mostrar_login():
    global vista_actual
    vista_actual = "login"
    ventana.minsize(400, 500)
    for widget in contenedor_principal.winfo_children():
        widget.destroy()

    # Contenedor superior derecho para unificar los botones ? y X
    frame_top_buttons = tk.Frame(contenedor_principal, bg=COLOR_SURFACE_CARD)
    frame_top_buttons.place(relx=1.0, rely=0.0, x=-10, y=10, anchor="ne")

    # Botón de Ayuda (?) - Login
    def mostrar_ayuda_login():
        messagebox.showinfo(
            "Ayuda - Inicio de Sesión",
            "Instrucciones de Login:\n\n"
            "1. Ingrese su Nombre de Usuario y Contraseña en los campos correspondientes.\n"
            "2. Presione 'INICIAR SESIÓN' para validar sus credenciales tradicionales.\n"
            "3. Alternativamente, si vinculó su rostro, presione 'Ingresar con Rostro 📷' para usar el escaneo facial.\n"
            "4. Si ha olvidado sus credenciales, haga clic en '¿Olvidaste tu contraseña?' para recuperarla mediante su pregunta de seguridad.\n"
            "Nota: El sistema se bloqueará por 3 minutos tras 3 intentos fallidos."
        )

    btn_help = tk.Button(frame_top_buttons, text="?", font=("Arial", 10, "bold"), bg=COLOR_INPUT_BG, fg="white", bd=0, width=3, height=1)
    btn_help.config(command=mostrar_ayuda_login)
    btn_help.pack(side="left", padx=1)

    btn_x = tk.Button(frame_top_buttons, text="X", font=("Arial", 10, "bold"), bg=COLOR_PRIMARY_RED, fg="white", bd=0, width=3, height=1, command=mostrar_lobby)
    btn_x.pack(side="left", padx=1)

    btn_settings.config(command=mostrar_lobby)

    contenedor_principal.place(relx=0.5, rely=0.5, anchor="center", width=500, height=530)

    lbl_title = tk.Label(contenedor_principal, text="AVATARS VS ROOKS", font=("Segoe UI", 18, "bold"), fg=COLOR_PRIMARY_RED, bg=COLOR_SURFACE_CARD)
    lbl_title.pack(pady=(20, 2))

    lbl_subtitle = tk.Label(contenedor_principal, text="Ingresa para comenzar la batalla", font=("Segoe UI", 9), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD)
    lbl_subtitle.pack(pady=(0, 25))

    frame_user = tk.Frame(contenedor_principal, bg=COLOR_SURFACE_CARD)
    frame_user.pack(fill="x", padx=30, pady=5)
    tk.Label(frame_user, text="Nombre de Usuario", font=("Segoe UI", 8), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(anchor="w", padx=5)
    entry_usuario = tk.Entry(frame_user, font=("Segoe UI", 11), bg=COLOR_INPUT_BG, fg=COLOR_TEXT_MAIN, bd=0, insertbackground=COLOR_TEXT_MAIN, highlightthickness=1, highlightbackground=COLOR_TEXT_SEC, highlightcolor=COLOR_PRIMARY_RED)
    entry_usuario.pack(fill="x", ipady=6, pady=3)

    frame_pass = tk.Frame(contenedor_principal, bg=COLOR_SURFACE_CARD)
    frame_pass.pack(fill="x", padx=30, pady=5)
    tk.Label(frame_pass, text="Contraseña (8-16 carac. / 1 Mayús. / 1 Núm.)", font=("Segoe UI", 8), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(anchor="w", padx=5)
    
    # Contenedor para password + ojito
    frame_p_input = tk.Frame(frame_pass, bg=COLOR_SURFACE_CARD)
    frame_p_input.pack(fill="x")
    entry_password = tk.Entry(frame_p_input, font=("Segoe UI", 11), bg=COLOR_INPUT_BG, fg=COLOR_TEXT_MAIN, bd=0, show="*", insertbackground=COLOR_TEXT_MAIN)
    entry_password.pack(side="left", fill="x", expand=True, ipady=6, pady=3)
    btn_ojito = tk.Button(frame_p_input, text="👁", bg=COLOR_INPUT_BG, fg=COLOR_TEXT_SEC, bd=0, command=lambda: toggle_password(entry_password))
    btn_ojito.pack(side="right", padx=5)

    btn_login = tk.Button(contenedor_principal, text="INICIAR SESIÓN", font=("Segoe UI", 11, "bold"), bg=COLOR_PRIMARY_RED, fg=COLOR_TEXT_MAIN, bd=0, cursor="hand2", command=lambda: intentar_login(entry_usuario, entry_password))
    btn_login.pack(fill="x", padx=30, ipady=8, pady=(20, 5))
    configuring_efecto_hover(btn_login, COLOR_PRIMARY_RED, "#ff3333")

    btn_facial = tk.Button(contenedor_principal, text="Ingresar con Rostro 📷", font=("Segoe UI", 9), bg=COLOR_SURFACE_CARD, fg=COLOR_TEXT_SEC, activebackground="#384459", activeforeground=COLOR_TEXT_MAIN, bd=1, relief="solid", cursor="hand2", highlightbackground=COLOR_TEXT_SEC, command=lambda: login_con_rostro(al_reconocer))
    btn_facial.pack(fill="x", padx=30, ipady=5, pady=5)
    configuring_efecto_hover(btn_facial, COLOR_SURFACE_CARD, "#3d4a60")

    tk.Label(contenedor_principal, text="¿Eres nuevo? Regístrate aquí", font=("Segoe UI", 8), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(pady=(15, 0))

    btn_go_register = tk.Button(contenedor_principal, text="REGISTRARSE", font=("Segoe UI", 10, "bold"), bg=COLOR_PRIMARY_RED, fg=COLOR_TEXT_MAIN, bd=0, cursor="hand2", command=mostrar_registro)
    btn_go_register.pack(fill="x", padx=30, ipady=6, pady=5)
    configuring_efecto_hover(btn_go_register, COLOR_PRIMARY_RED, "#ff3333")

    lbl_forgot = tk.Label(contenedor_principal, text="¿Olvidaste tu contraseña?", font=("Segoe UI", 9, "underline"), fg=COLOR_PRIMARY_RED, bg=COLOR_SURFACE_CARD, cursor="hand2")
    lbl_forgot.pack(pady=(5, 10))
    lbl_forgot.bind("<Button-1>", lambda e: recover_password())
    
    lbl_back_lobby = tk.Label(contenedor_principal, text="← Volver al Menú Principal", font=("Segoe UI", 8, "underline"), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD, cursor="hand2")
    lbl_back_lobby.pack(pady=2)
    lbl_back_lobby.bind("<Button-1>", lambda e: mostrar_lobby())
    
    maximizar_ventana(ventana)


# --- VISTA: REGISTRO ---

def mostrar_registro():
    global vista_actual
    vista_actual = "registro"
    ventana.minsize(500, 700)
    for widget in contenedor_principal.winfo_children():
        widget.destroy()

    contenedor_principal.place(relx=0.5, rely=0.5, anchor="center", width=460, height=680)

    # --- IMPLEMENTACIÓN DE CANVAS CON SCROLLBAR PARA EL FORMULARIO ---
    canvas = tk.Canvas(contenedor_principal, bg=COLOR_SURFACE_CARD, bd=0, highlightthickness=0)
    scrollbar = tk.Scrollbar(contenedor_principal, orient="vertical", command=canvas.yview)
    
    # El frame del contenido ahora es hijo del Canvas
    frame_contenido = tk.Frame(canvas, bg=COLOR_SURFACE_CARD)
    
    def actualizar_scroll(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        
    window_id = canvas.create_window((0, 0), window=frame_contenido, anchor="nw", width=440)
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Empaquetado de la estructura del scrollbar
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    frame_contenido.bind("<Configure>", actualizar_scroll)

    # Soporte para la ruedita del mouse (MouseWheel)
    def _on_mousewheel(event):
        if event.num == 4:  # Linux scroll up
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Linux scroll down
            canvas.yview_scroll(1, "units")
        else:  # Windows/MacOS
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # Vincular evento de la ruedita al entrar al área de registro
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_mousewheel)
    canvas.bind_all("<Button-5>", _on_mousewheel)

    # Funciones para bloquear y reactivar el scroll de forma dinámica
    def desactivar_scroll_global():
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    def reactivar_scroll_global():
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

    # Desenlazar el evento global cuando salimos de la pantalla de registro
    def _unbind_mousewheel(event=None):
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    # Asegurar que se remueva el evento al destruir el contenedor
    frame_contenido.bind("<Destroy>", _unbind_mousewheel)

    # Contenedor superior derecho para unificar los botones ? y X
    frame_top_buttons = tk.Frame(frame_contenido, bg=COLOR_SURFACE_CARD)
    frame_top_buttons.place(relx=1.0, rely=0.0, x=-10, y=10, anchor="ne")

    # Botón de Ayuda (?) - Registro
    def mostrar_ayuda_registro():
        messagebox.showinfo(
            "Ayuda - Creación de Cuenta",
            "Instrucciones de Registro:\n\n"
            "1. Complete todos sus datos básicos (Nombre, Fecha, Género, Username y Hobbies).\n"
            "2. Contraseña: Debe poseer entre 8 y 16 caracteres, incluir obligatoriamente al menos una letra MAYÚSCULA y al menos un NÚMERO.\n"
            "3. Hobbies: Puede marcar una combinación personalizada de hasta un máximo de 3 opciones.\n"
            "4. Pregunta de Seguridad: Elija una y provea una respuesta para la futura validación o recuperación de contraseña.\n"
            "5. Registro Facial: Escriba primero su Username, haga clic en 'Usar Foto 📷' y mire fijo al lente para capturar las 10 muestras.\n"
            "6. Tarjeta: Ingrese los 16 dígitos requeridos para su membresía.\n"
            "7. Vencimiento de Tarjeta: Indique la fecha de expiración mediante el selector.\n"
            "8. Términos: Haga clic obligatoriamente en 'Leer términos y condiciones' para habilitar la casilla 'Acepto'."
        )

    btn_help = tk.Button(frame_top_buttons, text="?", font=("Arial", 10, "bold"), bg=COLOR_INPUT_BG, fg="white", bd=0, width=3, height=1)
    btn_help.config(command=mostrar_ayuda_registro)
    btn_help.pack(side="left", padx=1)

    btn_x = tk.Button(frame_top_buttons, text="X", font=("Arial", 10, "bold"), bg=COLOR_PRIMARY_RED, fg="white", bd=0, width=3, height=1, command=lambda: [_unbind_mousewheel(), mostrar_login()])
    btn_x.pack(side="left", padx=1)

    tk.Label(frame_contenido, text="CREAR CUENTA", font=("Segoe UI", 14, "bold"), fg=COLOR_PRIMARY_RED, bg=COLOR_SURFACE_CARD).pack(pady=(8, 2))

    frame_photo = tk.Frame(frame_contenido, bg=COLOR_SURFACE_CARD)
    frame_photo.pack(fill="x", padx=25, pady=2)
    tk.Label(frame_photo, text="Foto de Perfil / Registro Facial", font=("Segoe UI", 8), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(anchor="w", padx=2)
    
    frame_inner_photo = tk.Frame(frame_photo, bg=COLOR_SURFACE_CARD)
    frame_inner_photo.pack(fill="x")
    
    lbl_photo_status = tk.Label(frame_inner_photo, text="Ningún rostro guardado", font=("Segoe UI", 8, "italic"), fg=COLOR_TEXT_SEC, bg=COLOR_INPUT_BG, anchor="w")
    lbl_photo_status.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 5))
    
    def ejecutar_captura():
        user = ent_user.get().strip()
        if registrar_rostro(user):
            lbl_photo_status.config(text="📷 ¡Rostro Vinculado!", fg="#00ff00")

    btn_photo = tk.Button(frame_inner_photo, text="Usar Foto 📷", font=("Segoe UI", 8, "bold"), bg=COLOR_INPUT_BG, fg=COLOR_TEXT_MAIN, bd=1, relief="solid", cursor="hand2", command=ejecutar_captura)
    btn_photo.pack(side="right", ipady=2, padx=2)
    configuring_efecto_hover(btn_photo, COLOR_INPUT_BG, COLOR_PRIMARY_RED)

    def crear_fila_input(label_text, is_password=False, placeholder=""):
        frame = tk.Frame(frame_contenido, bg=COLOR_SURFACE_CARD)
        frame.pack(fill="x", padx=25, pady=1)
        tk.Label(frame, text=label_text, font=("Segoe UI", 8), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(anchor="w", padx=2)
        
        if is_password:
            frame_p = tk.Frame(frame, bg=COLOR_SURFACE_CARD)
            frame_p.pack(fill="x")
            entry = tk.Entry(frame_p, font=("Segoe UI", 10), bg=COLOR_INPUT_BG, fg=COLOR_TEXT_MAIN, bd=0, show="*", insertbackground=COLOR_TEXT_MAIN)
            entry.pack(side="left", fill="x", expand=True, ipady=3, pady=1)
            btn_ojito = tk.Button(frame_p, text="👁", bg=COLOR_INPUT_BG, fg=COLOR_TEXT_SEC, bd=0, command=lambda: toggle_password(entry))
            btn_ojito.pack(side="right", padx=5)
        else:
            entry = tk.Entry(frame, font=("Segoe UI", 10), bg=COLOR_INPUT_BG, fg=COLOR_TEXT_MAIN, bd=0, insertbackground=COLOR_TEXT_MAIN, highlightthickness=1, highlightbackground=COLOR_TEXT_SEC, highlightcolor=COLOR_PRIMARY_RED)
            if placeholder:
                entry.insert(0, placeholder)
            entry.pack(fill="x", ipady=3, pady=1)
        return entry

    ent_nombre = crear_fila_input("Nombre y Apellidos")
    
    # --- FUNCIONES DE CONTROL PARA LOS POPUPS DE LOS CALENDARIOS ---
    def vincular_eventos_calendario(widget_dateentry):
        widget_dateentry.update_idletasks()
        
        # Acceso directo seguro a la ventana Toplevel interna del dropdown
        popup = widget_dateentry._top_cal
        
        # Al abrirse el popup: Desactivamos el scroll
        widget_dateentry.bind("<<DateEntrySelected>>", lambda e: reactivar_scroll_global(), add="+")
        popup.bind("<Map>", lambda e: desactivar_scroll_global(), add="+")
        popup.bind("<Unmap>", lambda e: reactivar_scroll_global(), add="+")
        
        # Si presiona BackSpace dentro del popup, cierra el calendario (cancela la selección)
        popup.bind("<BackSpace>", lambda e: widget_dateentry._toggle_drop_down(), add="+")

    # --- CALENDARIO: FECHA DE NACIMIENTO ---
    frame_fecha = tk.Frame(frame_contenido, bg=COLOR_SURFACE_CARD)
    frame_fecha.pack(fill="x", padx=25, pady=1)
    tk.Label(frame_fecha, text="Fecha de Nacimiento (Calendario)", font=("Segoe UI", 8), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(anchor="w", padx=2)
    ent_fecha = DateEntry(frame_fecha, font=("Segoe UI", 10), bg=COLOR_INPUT_BG, fg=COLOR_TEXT_MAIN, bd=0, 
                          headersbackground=COLOR_SURFACE_CARD, headersforeground=COLOR_TEXT_MAIN,
                          selectbackground=COLOR_PRIMARY_RED, selectforeground=COLOR_TEXT_MAIN,
                          background=COLOR_INPUT_BG, foreground=COLOR_TEXT_MAIN, date_pattern="dd/mm/yyyy",
                          style="Calendario.TCombobox")
    ent_fecha.pack(fill="x", ipady=3, pady=1)
    vincular_eventos_calendario(ent_fecha)

    frame_gen = tk.Frame(frame_contenido, bg=COLOR_SURFACE_CARD)
    frame_gen.pack(fill="x", padx=25, pady=1)
    tk.Label(frame_gen, text="Género", font=("Segoe UI", 8), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(anchor="w", padx=2)
    combo_genero = ttk.Combobox(frame_gen, values=["Masculino", "Femenino", "Otro"], state="readonly", font=("Segoe UI", 9))
    combo_genero.set("Masculino")
    combo_genero.pack(fill="x", pady=1)

    ent_user = crear_fila_input("Username")
    ent_p1 = crear_fila_input("Contraseña (8-16 carac., 1 Mayús., 1 Núm.)", is_password=True)
    ent_p2 = crear_fila_input("Confirmar Contraseña", is_password=True)

    frame_hob = tk.Frame(frame_contenido, bg=COLOR_SURFACE_CARD)
    frame_hob.pack(fill="x", padx=25, pady=2)
    tk.Label(frame_hob, text="Hobbies (Máximo 3)", font=("Segoe UI", 8), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(anchor="w", padx=2)
    box_hobbies = tk.Frame(frame_hob, bg=COLOR_BOX_BG, bd=1, relief="solid")
    box_hobbies.pack(fill="x", ipady=2, pady=1)

    hobbies_opciones = ["Música", "Dibujo", "Ajedrez", "Lectura", "Código", "Deportes"]
    dict_variables_hobbies = {}
    
    def validar_limite_hobbies(nombre_variable_cambiada):
        activos = sum(1 for v in dict_variables_hobbies.values() if v.get())
        if activos > 3:
            dict_variables_hobbies[nombre_variable_cambiada].set(False)

    for idx, hobby in enumerate(hobbies_opciones):
        var = tk.BooleanVar()
        dict_variables_hobbies[hobby] = var
        chk = tk.Checkbutton(box_hobbies, text=hobby, variable=var, font=("Segoe UI", 8), bg=COLOR_BOX_BG, fg=COLOR_TEXT_MAIN, selectcolor=COLOR_INPUT_BG, activebackground=COLOR_BOX_BG, activeforeground=COLOR_TEXT_MAIN, bd=0, command=lambda h=hobby: validar_limite_hobbies(h))
        chk.grid(row=idx//2, column=idx%2, sticky="w", padx=20, pady=1)

    frame_preg = tk.Frame(frame_contenido, bg=COLOR_SURFACE_CARD)
    frame_preg.pack(fill="x", padx=25, pady=1)
    tk.Label(frame_preg, text="Pregunta de Seguridad", font=("Segoe UI", 8), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(anchor="w", padx=2)
    
    preguntas_list = [
        "¿Cuál fue el primer nombre de su mascota?",
        "¿Cuál es el mejor equipo de fútbol de CR?",
        "¿Cuál es la mejor carrera de la UCR?"
    ]
    combo_pregunta = ttk.Combobox(frame_preg, values=preguntas_list, state="readonly", font=("Segoe UI", 9))
    combo_pregunta.set(preguntas_list[0])
    combo_pregunta.pack(fill="x", pady=1)
    
    ent_respuesta = tk.Entry(frame_preg, font=("Segoe UI", 10), bg=COLOR_INPUT_BG, fg=COLOR_TEXT_MAIN, bd=0, insertbackground=COLOR_TEXT_MAIN, highlightthickness=1, highlightbackground=COLOR_TEXT_SEC, highlightcolor=COLOR_PRIMARY_RED)
    ent_respuesta.insert(0, "Respuesta")
    
    def on_focus_in(event):
        if ent_respuesta.get() == "Respuesta":
            ent_respuesta.delete(0, tk.END)
            
    def on_focus_out(event):
        if ent_respuesta.get() == "":
            ent_respuesta.insert(0, "Respuesta")

    ent_respuesta.bind("<FocusIn>", on_focus_in)
    ent_respuesta.bind("<FocusOut>", on_focus_out)
    ent_respuesta.pack(fill="x", ipady=3, pady=(3, 1))

    # --- DATOS DE TARJETA ---
    frame_tarjeta = tk.Frame(frame_contenido, bg=COLOR_SURFACE_CARD)
    frame_tarjeta.pack(fill="x", padx=25, pady=1)
    tk.Label(frame_tarjeta, text="Datos de Tarjeta (Membresía $20)", font=("Segoe UI", 8), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(anchor="w", padx=2)

    tarjeta_digits = []          
    tarjeta_visible = [False]    
    tarjeta_actualizando = [False]

    frame_tarjeta_input = tk.Frame(frame_tarjeta, bg=COLOR_SURFACE_CARD)
    frame_tarjeta_input.pack(fill="x")

    ent_tarjeta = tk.Entry(frame_tarjeta_input, font=("Segoe UI", 10), bg=COLOR_INPUT_BG, fg=COLOR_TEXT_MAIN,
                           bd=0, insertbackground=COLOR_TEXT_MAIN, highlightthickness=1,
                           highlightbackground=COLOR_TEXT_SEC, highlightcolor=COLOR_PRIMARY_RED)
    ent_tarjeta.pack(side="left", fill="x", expand=True, ipady=3, pady=1)

    def tarjeta_get_real():
        return "".join(tarjeta_digits)

    def tarjeta_render():
        tarjeta_actualizando[0] = True
        digits = tarjeta_digits
        grupos = []
        for i in range(0, len(digits), 4):
            grupo = digits[i:i+4]
            if tarjeta_visible[0]:
                grupos.append("".join(grupo))
            else:
                grupos.append("*" * len(grupo))
        texto = "-".join(grupos)
        ent_tarjeta.delete(0, tk.END)
        ent_tarjeta.insert(0, texto)
        ent_tarjeta.icursor(tk.END)
        tarjeta_actualizando[0] = False

    def tarjeta_on_key(event):
        if tarjeta_actualizando[0]:
            return "break"

        if event.keysym in ("Left", "Right", "Home", "End", "Tab", "Shift_L", "Shift_R", "Control_L", "Control_R"):
            return

        if event.keysym in ("BackSpace", "Delete"):
            if tarjeta_digits:
                tarjeta_digits.pop()
            tarjeta_render()
            return "break"

        if event.char and event.char.isdigit():
            if len(tarjeta_digits) < 16:
                tarjeta_digits.append(event.char)
                tarjeta_render()
            return "break"

        return "break"

    def toggle_tarjeta():
        tarjeta_visible[0] = not tarjeta_visible[0]
        tarjeta_render()

    ent_tarjeta.bind("<Key>", tarjeta_on_key)
    ent_tarjeta.bind("<Control-v>", lambda e: "break")
    ent_tarjeta.bind("<Control-V>", lambda e: "break")

    btn_ojo_tarjeta = tk.Button(frame_tarjeta_input, text="👁", bg=COLOR_INPUT_BG, fg=COLOR_TEXT_SEC, bd=0, command=toggle_tarjeta)
    btn_ojo_tarjeta.pack(side="right", padx=5)

    _ent_tarjeta_original_get = ent_tarjeta.get
    def tarjeta_get_digits_formatted():
        digits = tarjeta_digits
        grupos = [("".join(digits[i:i+4])) for i in range(0, len(digits), 4)]
        return "-".join(grupos)
    ent_tarjeta.get = tarjeta_get_digits_formatted

    # --- CALENDARIO: VENCIMIENTO DE TARJETA ---
    frame_vencimiento = tk.Frame(frame_contenido, bg=COLOR_SURFACE_CARD)
    frame_vencimiento.pack(fill="x", padx=25, pady=1)
    tk.Label(frame_vencimiento, text="Vencimiento de la Tarjeta (Calendario)", font=("Segoe UI", 8), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD).pack(anchor="w", padx=2)
    ent_vencimiento = DateEntry(frame_vencimiento, font=("Segoe UI", 10), bg=COLOR_INPUT_BG, fg=COLOR_TEXT_MAIN, bd=0,
                                headersbackground=COLOR_SURFACE_CARD, headersforeground=COLOR_TEXT_MAIN,
                                selectbackground=COLOR_PRIMARY_RED, selectforeground=COLOR_TEXT_MAIN,
                                background=COLOR_INPUT_BG, foreground=COLOR_TEXT_MAIN, date_pattern="dd/mm/yyyy",
                                style="Calendario.TCombobox")
    ent_vencimiento.pack(fill="x", ipady=3, pady=1)
    vincular_eventos_calendario(ent_vencimiento)

    # --- TÉRMINOS ---
    var_leido = tk.BooleanVar(value=False)
    var_terminos = tk.BooleanVar(value=False)

    def abrir_terminos():
        webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        var_leido.set(True)
        chk_terms.config(state="normal")

    frame_terms = tk.Frame(frame_contenido, bg=COLOR_SURFACE_CARD)
    frame_terms.pack(pady=5)
    
    btn_link = tk.Button(frame_terms, text="Leer términos y condiciones", font=("Segoe UI", 8, "underline"), fg=COLOR_PRIMARY_RED, bg=COLOR_SURFACE_CARD, bd=0, cursor="hand2", command=abrir_terminos)
    btn_link.pack(side="left")
    
    chk_terms = tk.Checkbutton(frame_terms, text="Acepto", variable=var_terminos, state="disabled", font=("Segoe UI", 8), bg=COLOR_SURFACE_CARD, fg=COLOR_TEXT_SEC, selectcolor=COLOR_INPUT_BG)
    chk_terms.pack(side="left", padx=5)

    def ejecutar_registro_final():
        user = ent_user.get().strip()
        p1 = ent_p1.get()
        p2 = ent_p2.get()
        nombre = ent_nombre.get().strip()
        fecha = ent_fecha.get().strip()
        genero = combo_genero.get()
        pregunta = combo_pregunta.get()
        respuesta = ent_respuesta.get().strip()
        tarjeta = ent_tarjeta.get().strip()
        vencimiento_tarjeta = ent_vencimiento.get().strip()
        
        hobbies_seleccionados = [h for h, v in dict_variables_hobbies.items() if v.get()]

        if not user or not p1 or not nombre or not respuesta or respuesta == "Respuesta" or len(tarjeta.replace("-", "").replace(" ", "")) < 16:
            messagebox.showwarning("Campos incompletos", "Por favor, completa todos los campos correctamente.")
            return

        if p1 != p2:
            messagebox.showerror("Error", "Las contraseñas de combate no coinciden.")
            return

        if not (8 <= len(p1) <= 16) or not re.search(r"[A-Z]", p1) or not re.search(r"[0-9]", p1):
            messagebox.showerror("Contraseña Débil", "La contraseña debe tener de 8 a 16 caracteres, incluir al menos una letra Mayúscula y al menos un Número.")
            return

        if not var_terminos.get():
            messagebox.showwarning("Términos", "Debes leer y aceptar los términos para entrar.")
            return

        exito = guardar_nuevo_usuario(user, p1, nombre, fecha, genero, hobbies_seleccionados, pregunta, respuesta, tarjeta, vencimiento_tarjeta)
        if not exito:
            messagebox.showerror("Error", "Ese Nombre de Usuario ya está en uso.")
            return

        messagebox.showinfo("¡REGISTRADO!", "¡Registrado con éxito! Perfil de gladiador guardado.")
        _unbind_mousewheel()
        mostrar_login()

    btn_submit = tk.Button(frame_contenido, text="COMPLETAR REGISTRO", font=("Segoe UI", 11, "bold"), bg=COLOR_PRIMARY_RED, fg=COLOR_TEXT_MAIN, bd=0, cursor="hand2", command=ejecutar_registro_final)
    btn_submit.pack(fill="x", padx=25, ipady=5, pady=(5, 15))
    configuring_efecto_hover(btn_submit, COLOR_PRIMARY_RED, "#ff3333")

    lbl_back = tk.Label(frame_contenido, text="← Volver al inicio de sesión", font=("Segoe UI", 9, "underline"), fg=COLOR_TEXT_SEC, bg=COLOR_SURFACE_CARD, cursor="hand2")
    lbl_back.pack(pady=(2, 15))
    lbl_back.bind("<Button-1>", lambda e: [_unbind_mousewheel(), mostrar_login()])
    
    maximizar_ventana(ventana)


# --- CONFIGURACIÓN DE LA RAÍZ ---

ventana = tk.Tk()
ventana.title("Avatars VS Rooks - Sistema Central")
ventana.configure(bg=COLOR_DEEP_BG)

style = ttk.Style()
style.theme_use('clam')
style.configure("TCombobox", fieldbackground=COLOR_INPUT_BG, background=COLOR_SURFACE_CARD, foreground=COLOR_TEXT_MAIN, arrowcolor=COLOR_TEXT_MAIN, bordercolor=COLOR_TEXT_SEC)
style.map("TCombobox", fieldbackground=[("readonly", COLOR_INPUT_BG)], foreground=[("readonly", COLOR_TEXT_MAIN)])

# --- ESTILO PARA LOS CALENDARIOS ---
style.layout("Calendario.TCombobox", [
    ('Combobox.field', {
        'sticky': 'nswe', 
        'children': [
            ('Combobox.padding', {
                'sticky': 'nswe', 
                'children': [
                    ('Combobox.textarea', {'sticky': 'nswe'})
                ]
            })
        ]
    })
])
style.configure("Calendario.TCombobox", fieldbackground=COLOR_INPUT_BG, background=COLOR_SURFACE_CARD, foreground=COLOR_TEXT_MAIN, bordercolor=COLOR_TEXT_SEC)

ventana.resizable(True, True)
maximizar_ventana(ventana)

# Variable global para registrar la pantalla actual de la aplicación
vista_actual = "lobby"

# Función inteligente para capturar el BackSpace sin romper la edición de los Entry
def manejar_backspace(event):
    if isinstance(event.widget, tk.Entry):
        return
    
    if vista_actual == "personalizacion":
        mostrar_lobby()
    elif vista_actual == "login":
        mostrar_lobby()
    elif vista_actual == "registro":
        mostrar_login()

# Vincular la tecla BackSpace a nivel de ventana global de forma segura
ventana.bind("<BackSpace>", manejar_backspace)

btn_settings = tk.Button(ventana, text="⚙", font=("Arial", 16), bg=COLOR_PRIMARY_RED, fg=COLOR_TEXT_MAIN, activebackground="#cc1111", activeforeground=COLOR_TEXT_MAIN, bd=0, width=2, height=1, cursor="hand2")
btn_settings.place(x=25, y=25)

lbl_music = tk.Label(ventana, text="🎵 Melodía: Tensión y Adrenalina", font=("Segoe UI", 8), fg=COLOR_TEXT_SEC, bg=COLOR_DEEP_BG)
lbl_music.place(relx=0.0, rely=1.0, x=25, y=-25, anchor="sw")

lbl_lang = tk.Label(ventana, text="ES | EN", font=("Segoe UI", 9, "bold"), fg=COLOR_TEXT_SEC, bg=COLOR_DEEP_BG)
lbl_lang.place(relx=1.0, rely=1.0, x=-25, y=-25, anchor="se")

contenedor_principal = tk.Frame(ventana, bg=COLOR_SURFACE_CARD, bd=1, relief="solid", highlightthickness=0)
contenedor_principal.config(highlightbackground="#4a5568") 

mostrar_lobby()

ventana.after(100, lambda: maximizar_ventana(ventana))
ventana.mainloop()