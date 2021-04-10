# import output_file
# from multiprocess import Pool

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    #with Pool(workers) as p:
    hist = model.fit_generator(
            tg,
            steps_per_epoch=len(tg),
            validation_data=vg,
            validation_steps=8,
            epochs=epochs,
            use_multiprocessing=use_multiprocessing,  # you have to train the model on GPU in order to this to be benefitial
            workers=workers,  # you have to train the model on GPU in order to this to be benefitial
            verbose=1,
            callbacks=[checkpoint],
        )

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title("loss")
    ax[0].plot(hist.epoch, hist.history["loss"], label="Train loss")
    ax[0].plot(hist.epoch, hist.history["val_loss"], label="Validation loss")
    ax[1].set_title("acc")
    ax[1].plot(hist.epoch, hist.history["f1"], label="Train F1")
    ax[1].plot(hist.epoch, hist.history["val_f1"], label="Validation F1")
    ax[0].legend()
    ax[1].legend()
