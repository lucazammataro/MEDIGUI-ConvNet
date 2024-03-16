#!/usr/bin/env python
# coding: utf-8


'''

MEDIGUI-ConvNet (Medical Imaging Convolutional Neural Network with Graphic User Interface)
Luca Zammataro, LunanFoldomics LLC, 2024

'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import History
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, Label, Layout, Output, Tab
from ipyfilechooser import FileChooser
from IPython.display import display
import io
import shutil
from contextlib import redirect_stdout
from tqdm.notebook import tqdm  # Importa tqdm per la barra di avanzamento
from datetime import datetime
import pickle
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

tf.config.set_visible_devices([], 'GPU')


# Definisci una funzione per calcolare il massimo all'interno di ciascun elenco
def max_in_list(lst):
    return max(lst)



def LoadImageArchive(path):
    
    with open(path, 'rb') as handle:
        df_dataset = pickle.load(handle)

    # Applica la funzione alla colonna 'X' per ottenere il massimo in ogni lista
    max_values = df_dataset['X'].apply(max_in_list)

    # Trova il massimo tra tutti i valori ottenuti
    max_value = max_values.max()

    # Normalizza i valori della colonna 'X' dividendo ogni valore per il massimo trovato
    df_dataset['X_normalized'] = df_dataset['X'].apply(lambda x: [val / max_value for val in x])

    X = np.stack(df_dataset['X_normalized'])
    y = np.int64(df_dataset['y'])    
    Y = to_categorical(y, len(set(df_dataset['y'].values)))

    X = X.reshape((X.shape[0], np.int64(np.sqrt(X.shape[1])), np.int64(np.sqrt(X.shape[1]))))
    
    return X, Y


# In[21]:


def LoadImageArchiveAndIntegrate(path):
    
    with open(path, 'rb') as handle:
        CVX = pickle.load(handle)

    # Applica la funzione alla colonna 'X' per ottenere il massimo in ogni lista
    max_values = CVX['X'].apply(max_in_list)

    # Trova il massimo tra tutti i valori ottenuti
    max_value = max_values.max()

    # Normalizza i valori della colonna 'X' dividendo ogni valore per il massimo trovato
    CVX['X_normalized'] = CVX['X'].apply(lambda x: [val / max_value for val in x])

    X = np.stack(CVX['X_normalized'])
    y = np.int64(CVX['y'])    
    Y = to_categorical(y, 5)

    
    # Genomic info integration for 100X100
    
    n = int(X.shape[1]/2)

    for i in range(len(y)):
        if y[i] == 0:
            X[i][n]= 0.7
        if y[i] == 1:
            X[i][n]= 1.0
        if y[i] == 2:
            X[i][n]= 0.0
        if y[i] == 3:
            X[i][n]= 0.5
        if y[i] == 4:
            X[i][n]= 0.5
    
    
    X = X.reshape((X.shape[0], np.int64(np.sqrt(X.shape[1])), np.int64(np.sqrt(X.shape[1]))))
    
    return X, Y


# In[22]:


def DisplayImage(X, Y, img_index, classes):
    
    img_index = img_index
    img = X[img_index]
    
    x = len(classes)
    def list_my_classes(x):
        return [i for i in range(x)]

    classes_list = list_my_classes(x)
    classes_keys = classes_list
    classes_values = classes
    tuple_lists = zip(classes_keys, classes_values)
    dict_classes = dict(tuple_lists)    
    

    # Reshape the image to a 1x28x28 array (as expected by the model)
    img = np.expand_dims(img, axis=0)

    plt.imshow(X[img_index], cmap='gray')
    print('Archive size:', X.shape)  
    print('Image size:', str(X.shape[1])+'X'+str(X.shape[2])+'px')  
    print('Displaying Image ID:', img_index)
    print('Class:', dict_classes[np.argmax(Y[img_index])])
    

    plt.show()


# In[23]:


def splitDataset(X, Y, test_size, random_state):
    # Dividi gli array X e Y in set di training e test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, Y_train, Y_test


# In[24]:


def defineModel(X, Y, l1, l2):
    # Definizione del modello di rete neurale convoluzionale (CNN) con regolarizzazione L2
    model = models.Sequential([
        layers.Conv2D(44, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),  # Aggiunta di regolarizzazione L2
        layers.Dense(Y.shape[1], activation='softmax')
    ])



    # Compilazione del modello
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    #model.summary()
    
    return model


# In[25]:


def trainCNN(X_train, X_test, Y_train, Y_test, epochs, batch_size):
    # Assicurati di avere i tuoi dati di addestramento (train_images) e le relative etichette (train_labels)
    # Assicurati di avere anche i tuoi dati di test (test_images) e le relative etichette (test_labels)

    # Addestramento del modello
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))

    # Valutazione del modello
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(f'Test accuracy: {test_acc}')

    # Visualizzazione delle curve di apprendimento

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    
    return model


# In[26]:


def testCNN(X, Y, model, classes, img_index):
    # TESTING

    
    x = len(classes)
    def list_my_classes(x):
        return [i for i in range(x)]

    classes_list = list_my_classes(x)
    #print(classes_list)    
    
    classes_keys = classes_list
    classes_values = classes
    tuple_lists = zip(classes_keys, classes_values)
    dict_classes = dict(tuple_lists)    
    
    #img_index = np.random.randint(0, len(X))
    img = X[img_index]

    # Reshape the image to a 1x28x28 array (as expected by the model)
    img = np.expand_dims(img, axis=0)

    # Make a prediction using the model
    logits = model.predict(img)

    # Get the predicted class (the index of the maximum logit)
    predicted_class = np.argmax(logits)

    # Print the predicted class and show the image
    print('Archive size:', X.shape)  
    print('Image size:', str(X.shape[1])+'X'+str(X.shape[2])+'px')  
    print('Displaying Image ID:', img_index)
    print('True Class:', dict_classes[np.argmax(Y[img_index])])
    print(f"Predicted Class: {dict_classes[predicted_class]}")
    
    #print(f"Predicted Class: {predicted_class}")
    #print('True Class:', np.argmax(Y[img_index]))
    
    plt.imshow(X[img_index], cmap='gray')

    plt.show()


# In[27]:


def saveModel(model_path):
    tf.keras.models.save_model(model, model_path)


# In[28]:


def loadModel(model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model


# In[29]:


output = Output()
model = None
X = np.array([[0],[0],[0]])

# Creare un widget di testo per visualizzare i messaggi
output_text = widgets.Textarea(description="Log:")
# Impostare la larghezza e l'altezza desiderate
output_text.layout.width = '800px'
output_text.layout.height = '550px'


# Funzione per stampare un messaggio nel widget di testo
def print_message(message):
    output_text.value += message + '\n'



# Definire la funzione DisplayTest
def DisplayTest(X, Y, Image_ID):
    global output
    with output:
        testCNN(X=X, Y=Y, model=model, classes=classes, img_index=Image_ID)        
    output.clear_output(wait=True)       


# Definire una funzione per l'attivazione di DisplayTest
def on_button_click_DisplayTest(b):
    DisplayTest(X, Y, int(Image_ID_slider_widget.value)) 


# Creare il widget Text
Image_ID_slider_widget = widgets.IntSlider(value=0, min=0, max=X.shape[1]) 
# Creare un pulsante interattivo
display_image_button = widgets.Button(description="Display a Test Image")
display_image_button.on_click(on_button_click_DisplayTest)
# Nascondere inizialmente il widget Image_ID_slider_widget e il pulsante display_image_button
Image_ID_slider_widget.layout.visibility = 'hidden'
display_image_button.layout.visibility = 'hidden'

# Funzione per gestire il clic sul pulsante di azione
def on_button_click_Load_Dataset(b):
    global X, Y, classes, output, directory_path, filename
    directory_path = dataset_file_chooser.selected_path   
    filename = dataset_file_chooser.selected_filename
    try:
        # Creare una barra di progresso
        progress_bar = widgets.IntProgress(
            value=50,
            min=0,
            max=100,
            description='Loading...',
            style={'bar_color': 'lightblue'}
        )

        display(progress_bar)        
        X, Y = LoadImageArchive(directory_path+'/'+filename)
        #X, Y = LoadImageArchiveAndIntegrate(directory_path+'/'+filename)      

        # Load the classes file
        df_classes = pd.read_csv(directory_path+'/'+filename.replace('pickle','')+'classes.tsv', sep='\t')
        classes = list(df_classes['class'].values)        

        # Aggiorna la barra di progresso
        progress_bar.value += 50       

        print_message("\n")
        print_message(f"Dataset: {directory_path+'/'+filename.replace('pickle','')} successfully loaded!")
        print_message(f"Class file: {directory_path+'/'+filename.replace('pickle','')+'classes.tsv'} successfully loaded!")     
        print_message("\n")        
        print_message(f"Dataset size: {X.shape[0]} items")  
        print_message(f"Image size: {str(X.shape[1])+'X'+str(X.shape[2])+'px'}")  


        progress_bar.layout.visibility = 'hidden'
        Image_ID_slider_widget.layout.visibility = 'hidden'
        display_image_button.layout.visibility = 'hidden'
        output.clear_output(wait=True)

        # Mostra il widget Image_ID_slider_widget e il pulsante display_image_button
        Image_ID_slider_widget.max=X.shape[0]-1

    except Exception as e:
        print(f"Error in loading dataset: {e}")



# Crea una VBox vuota per il plot finale
plot_vbox = VBox([])

# Definire la funzione per avviare il training del modello
def on_button_click_Train_Model(b):
    global model, X_train, X_test, Y_train, Y_test, plot_vbox, output

    Image_ID_slider_widget.layout.visibility = 'hidden'
    display_image_button.layout.visibility = 'hidden'
    #output.clear_output(wait=True)


    try:
        # Suddividere il dataset in set di training e di test
        X_train, X_test, Y_train, Y_test = splitDataset(X=X, Y=Y, test_size=0.2, random_state=42)

        # Verifica le nuove forme degli array
        print_message("\n")
        print_message("Training and Testing sets have been randomly splitted.")
        print_message("\n")
        print_message("Training set:")
        print_message(f"X_train shape: {X_train.shape}")
        print_message(f"Y_train shape: {Y_train.shape}")
        print_message("\n")        
        print_message("Testing set:")
        print_message(f"X_test shape: {X_test.shape}")
        print_message(f"Y_test shape: {Y_test.shape}")
        print_message("\n")            

        # Definire il modello
        model = defineModel(X=X, Y=Y, l1=l1_widget.value, l2=l2_widget.value)

        # Stampare il summary del modello
        print_message("Model Summary:")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            model.summary()
        summary_text = buffer.getvalue()
        print_message(summary_text)

        # Avviare il training del modello con la barra di avanzamento tqdm
        print_message("Training...")    

        history = History()  # Crea un oggetto History per registrare la storia dell'addestramento


        with tqdm(total=epochs_widget.value) as pbar:  # Inizializza la barra di avanzamento
            for _ in range(epochs_widget.value):
                # Addestramento del modello per una singola epoca con il callback History
                model.fit(X_train, Y_train, epochs=1, batch_size=batch_size_widget.value, validation_data=(X_test, Y_test), verbose=0, callbacks=[history])
                pbar.update(1)  # Incrementa la barra di avanzamento di 1

        # Close the pbar after the learning is finished, and clean all.
        # Setta la posizione della barra di avanzamento al di fuori dello schermo
        pbar.set_postfix_str("")
        pbar.set_description_str("")
        pbar.refresh()


        # Valutazione del modello
        test_loss, test_acc = model.evaluate(X_test, Y_test)
        print_message(f'Test accuracy: {test_acc}')

        # Visualizzazione delle curve di apprendimento
        history_dict = history.history  # Ottieni la storia dell'addestramento
        plt.plot(history_dict['accuracy'], label='accuracy')
        plt.plot(history_dict['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        # Creare un widget che contenga il grafico
        plot_widget = widgets.Output()
        with plot_widget:
            plt.show()

        # Aggiorna il contenuto di plot_vbox con il nuovo widget del grafico
        plot_vbox.children = [plot_widget]        

        # Crea una nuova VBox per il plot finale
        empty_vbox = VBox([])        

        # Creare una nuova VBox per il plot finale se plot_vbox non è None, altrimenti usa empty_vbox
        #plot_vbox = plot_vbox if plot_vbox is not None else empty_vbox

        #plot_vbox = VBox([plt.gcf()])

        print_message("\n")
        print_message("Model successfully trained!")

        # Automatically save the trained model to disk, with date.
        # Get date and current hour
        now = datetime.now()

        # Formattare la data e l'ora come stringa nel formato desiderato
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Salvataggio del modello con la data
        saveModel(directory_path+'/Models/'+filename.replace('pickle','')+timestamp+'.model')

        print_message("\n")
        print_message(f"Model saved in: {directory_path+'/Models/'+filename.replace('pickle','')+timestamp+'.model'}")

        train_df = pd.DataFrame([list(X_train.reshape(X_train.shape[0], -1)), list(np.argmax(Y_train, axis=1))]).T
        test_df = pd.DataFrame([list(X_test.reshape(X_test.shape[0], -1)), list(np.argmax(Y_test, axis=1))]).T
        train_df.columns = ['X', 'y']
        test_df.columns = ['X', 'y']

        # Salvataggio dei DataFrame di training and testing in file pickle separati
        train_df.to_pickle(directory_path+'/'+filename.replace('pickle','')+timestamp+'.training.dataset.pickle')
        shutil.copyfile(directory_path+'/'+filename.replace('pickle','')+'classes.tsv', directory_path+'/'+filename.replace('pickle','')+timestamp+'.training.dataset.classes.tsv')
        test_df.to_pickle(directory_path+'/'+filename.replace('pickle','')+timestamp+'.testing.dataset.pickle')
        shutil.copyfile(directory_path+'/'+filename.replace('pickle','')+'classes.tsv', directory_path+'/'+filename.replace('pickle','')+timestamp+'.testing.dataset.classes.tsv')        
        
        print_message(f"Training and testing datasets saved in: {directory_path}")
        

    except Exception as e:
        print_message(f"Error during model training: {e}")


# Definire la funzione per avviare il testing del modello

def on_button_click_Load_Model(b):
    global model    
    directory_model_path = model_file_chooser.selected_path

    # Controlla se il dataset è stato caricato
    if X.shape[1] == 1 :
        print_message("Error: Load the dataset, before loading the model.")
        #print("Errore: Prima di caricare il modello, devi caricare il dataset.")
        return

    try:
        #model = tf.keras.models.load_model(directory_model_path)
        model = loadModel(directory_model_path)  
        print_message("\n")        
        print_message(f"Model: {directory_model_path} successfully loaded!")

        # Mostra il widget Image_ID_slider_widget e il pulsante display_image_button
        Image_ID_slider_widget.layout.visibility = 'visible'
        display_image_button.layout.visibility = 'visible'   


    except Exception as e:
        print_message("\n")        
        print_message(f"Error in loading model: {e}")






# DATASET LOADING CONTROLS

# Creare il widget FileChooser per selezionare una directory
dataset_file_chooser = FileChooser(filter_pattern='*.pickle', title='Select an image dataset')

# Creare un pulsante per avviare l'azione
load_dataset_button = widgets.Button(description="Load dataset")

# Collegare la funzione on_button_click all'evento di clic del pulsante
load_dataset_button.on_click(on_button_click_Load_Dataset)



# MODEL TRAINING CONTROLS

# Creare i widget per controllare i parametri di training del modello
epochs_widget = widgets.IntSlider(value=30, min=1, max=100, step=1, description='Epochs:')
batch_size_widget = widgets.IntSlider(value=32, min=1, max=128, step=1, description='Batch Size:')
l1_widget = widgets.FloatSlider(value=0.001, min=0, max=0.1, step=0.001, description='L1:')
l2_widget = widgets.FloatSlider(value=0.001, min=0, max=0.1, step=0.001, description='L2:')


# Creare un pulsante per avviare il training del modello
train_model_button = widgets.Button(description="Train Model")

# Collegare la funzione on_button_click_Train_Model all'evento di clic del pulsante
train_model_button.on_click(on_button_click_Train_Model)

# Mettere insieme i widget e i pulsanti all'interno di una VBox
training_controls_box = VBox([epochs_widget, batch_size_widget, l1_widget, l2_widget, train_model_button])


# MODEL TESTING CONTROLS

# Creare il widget FileChooser per selezionare una directory
model_file_chooser = FileChooser(title='Select a saved CNN model')

# Creare un pulsante per avviare l'azionetitle='\n')
# Creare un pulsante per avviare l'azione
load_model_button = widgets.Button(description="Load a saved model")

# Collegare la funzione on_button_click all'evento di clic del pulsante
load_model_button.on_click(on_button_click_Load_Model)


# ABOUT TAB

author = 'Luca Zammataro, 2024'
algorithm = 'MEDIGUI-ConvNet (Medical Imaging Convolutional Neural Network with Graphic User Interface)'

# Crea la tabella con i tuoi dati
table_html = f"""
<table>
  <tr>
    <th colspan="5" style="text-align:center;">Algorithm: {algorithm}</th>
  </tr>
  <tr>
    <td colspan="5"></td>
  </tr>
  <tr>
    <th style="text-align:left;">Author:</th>
    <th></th>
    <th></th>
    <th></th>
    <th style="text-align:left;">{author}</th>
  </tr>
</table>
"""

# Crea il widget HTML con la tabella
about_table_widget = widgets.HTML(value=table_html)

# Includere i controlli in un unico Tab
tab_contents = ['Load Dataset', 'Model Training', 'Learning Plot', 'Model Testing', 'About']


# Includere i controlli in un unico Tab
tab_contents = ['Load Dataset', 'Model Training', 'Learning Plot', 'Model Testing', 'About']


children = [
            VBox([dataset_file_chooser, load_dataset_button]), 
            VBox([training_controls_box]),
            VBox([plot_vbox]),
            VBox([model_file_chooser, load_model_button]),
            VBox([about_table_widget])
           ]

tab = Tab()
tab.children = children
for i in range(len(children)):
    tab.set_title(i, tab_contents[i])

# Visualizzare il widget Tab
display(tab)
display(HBox([Image_ID_slider_widget, display_image_button]))
display(HBox([output_text, output]))






