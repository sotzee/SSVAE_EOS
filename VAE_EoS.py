#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE_EoS with easy save/load:
- Serializable Sampling layer
- build_encoder / build_decoder / build_auto
- save_all / load_all (models, scalers, config, ckpts)
"""

import os, json, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib  # pip install joblib

tf.config.set_visible_devices([], 'GPU')
# ----------------------------
# Serializable Sampling layer
# ----------------------------
@tf.keras.utils.register_keras_serializable(package="VAE")
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps
    def get_config(self):
        return {}

@tf.keras.utils.register_keras_serializable(package="VAE")
class SliceFirstN(layers.Layer):
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = int(n)

    def call(self, x):
        return x[:, : self.n]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n": self.n})
        return cfg


@tf.keras.utils.register_keras_serializable(package="VAE")
class SliceLastN(layers.Layer):
    def __init__(self, n_last, **kwargs):
        super().__init__(**kwargs)
        self.n_last = int(n_last)

    def call(self, x):
        return x[:, -self.n_last :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_last": self.n_last})
        return cfg


@tf.keras.utils.register_keras_serializable(package="VAE")
class NegSoftplus(layers.Layer):
    def call(self, x):
        return -tf.nn.softplus(x)

    def get_config(self):
        return super().get_config()

class VAE_EoS:
    def __init__(self):
        # raw data
        self.df_data_obs = None
        self.df_cs2 = None
        self.df_boundary = None

        # splits (scaled)
        self.X_train_scaled = None
        self.X_val_scaled   = None
        self.X_test_scaled  = None
        self.y_train_obs_scaled = None
        self.y_val_obs_scaled   = None
        self.y_test_obs_scaled  = None

        # scalers
        self.X_scaler = None
        self.y_scaler = None

        # dims / hypers
        self.input_dim = None
        self.latent_dim_supervised = None
        self.latent_dim_variational = None
        self.eta = None
        self.kappa = None
        self.learning_rate = None
        self.batch_size = None
        self.epochs = None
        self.patience = None

        # models / optimizer / ckpt
        self.encoder = None   # tf.keras.Model
        self.decoder = None   # tf.keras.Model
        self.auto    = None   # tf.keras.Model (end-to-end)
        self.optimizer = None
        self._ckpt = None
        self._ckpt_manager = None

    # -------- Data prep --------
    def inputs(self,data_path):
        from read_data import read_data
        X_train, X_val, X_test, y_train_obs, y_val_obs, y_test_obs, df_data_obs, df_cs2, df_boundary = read_data(data_path)

        # ----------------------------
        # 4. Normalize Inputs
        # ----------------------------
        self.X_scaler = StandardScaler().fit(X_train)
        self.y_scaler = StandardScaler().fit(y_train_obs)

        self.X_train_scaled = self.X_scaler.transform(X_train).astype('float32')
        self.X_val_scaled   = self.X_scaler.transform(X_val).astype('float32')
        self.X_test_scaled  = self.X_scaler.transform(X_test).astype('float32')

        self.y_train_obs_scaled = self.y_scaler.transform(y_train_obs).astype('float32')
        self.y_val_obs_scaled   = self.y_scaler.transform(y_val_obs).astype('float32')
        self.y_test_obs_scaled  = self.y_scaler.transform(y_test_obs).astype('float32')
        return self

    def hypers(self):
        # ----------------------------
        # 5. Hyperparameters
        # ----------------------------
        self.input_dim = self.X_train_scaled.shape[1]
        self.latent_dim_supervised = self.y_train_obs_scaled.shape[1]
        self.latent_dim_variational = 4

        self.eta = 0.001
        self.kappa = 10.0
        self.learning_rate = 1e-4
        self.batch_size = 64
        self.epochs = 500
        self.patience = 10
        return self

    # -------- Model builders --------
    def build_encoder(self):
        enc_in = tf.keras.Input(shape=(self.input_dim,), name='encoder_input')
        activation = 'swish' #used to be 'relu'
        x = layers.Dense(128, activation=activation)(enc_in)
        x = layers.Dense(128, activation=activation)(x)
        x = layers.Dense(64, activation=activation)(x)
        x = layers.Dense(64, activation=activation)(x)

        # deterministic supervised latents
        mu_sup = layers.Dense(self.latent_dim_supervised, name='mu_supervised')(x)

        # variational latents
        mu_var = layers.Dense(self.latent_dim_variational, name='mu_variational')(x)
        log_var = layers.Dense(self.latent_dim_variational, name='log_var_variational')(x)
        z_var = Sampling()([mu_var, log_var])

        # full latent
        z_full = layers.Concatenate(name='z_concat')([mu_sup, z_var])

        self.encoder = tf.keras.Model(
            enc_in, [mu_sup, mu_var, log_var, z_full], name="encoder"
        )
        return self

    # def build_decoder(self):
    #     dec_in = tf.keras.Input(shape=(self.latent_dim_supervised+self.latent_dim_variational,), name='decoder_input')
    #     xd = layers.Dense(64, activation='relu')(dec_in)
    #     xd = layers.Dense(64, activation='relu')(xd)
    #     xd = layers.Dense(128, activation='relu')(xd)
    #     dec_out = layers.Dense(self.input_dim, activation='linear')(xd)
    #     self.decoder = tf.keras.Model(dec_in, dec_out, name="decoder")
    #     return self
    def build_decoder(self):
        n_first = int(self.input_dim - 6)
    
        dec_in = tf.keras.Input(
            shape=(self.latent_dim_supervised + self.latent_dim_variational,),
            name="decoder_input"
        )
        activation = 'swish' #used to be 'relu'
        xd = layers.Dense(64, activation=activation)(dec_in)
        xd = layers.Dense(64, activation=activation)(xd)
        xd = layers.Dense(128, activation=activation)(xd)
        xd = layers.Dense(128, activation=activation)(xd)
    
        xd = layers.Dense(self.input_dim, activation="linear", name="z_raw")(xd)

        #dec_out = xd
        
        xd_first = SliceFirstN(n_first, name="y_first")(xd)
        xd_last6 = SliceLastN(6, name="y_last6")(xd)
    
        xd_first_valid = NegSoftplus(name="y_first_leq0")(xd_first)
        # xd_first_valid = xd_first
    
        dec_out = layers.Concatenate(axis=-1, name="y_valid")([xd_first_valid, xd_last6])
    
        self.decoder = tf.keras.Model(dec_in, dec_out, name="decoder")
        return self




    def build_auto(self):
        x_in = tf.keras.Input(shape=(self.input_dim,), name="x_in")
        mu_sup, mu_var, log_var, z = self.encoder(x_in)
        x_out = self.decoder(z)
        # Keep all useful outputs for downstream use
        self.auto = tf.keras.Model(
            x_in, [x_out, mu_sup, mu_var, log_var, z], name="vae_eos"
        )
        return self

    def loop(self, outdir="artifacts"):
        os.makedirs(outdir, exist_ok=True)
        print('latent_dim_variational=',self.latent_dim_variational)
        print('eta=', self.eta)
        print('kappa=', self.kappa)

        self.build_encoder().build_decoder().build_auto()

        # Optimizer / losses / datasets
        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        mse_loss_fn = tf.keras.losses.MeanSquaredError()

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_train_scaled, self.y_train_obs_scaled)
        ).shuffle(1024).batch(self.batch_size)
        val_dataset   = tf.data.Dataset.from_tensor_slices(
            (self.X_val_scaled, self.y_val_obs_scaled)
        ).batch(self.batch_size)
        test_dataset  = tf.data.Dataset.from_tensor_slices(
            (self.X_test_scaled, self.y_test_obs_scaled)
        ).batch(self.batch_size)

        # Checkpoint (captures optimizer slots; useful if you continue training)
        self._ckpt = tf.train.Checkpoint(
            encoder=self.encoder, decoder=self.decoder, optimizer=self.optimizer
        )
        self._ckpt_manager = tf.train.CheckpointManager(
            self._ckpt, os.path.join(outdir, "ckpts"), max_to_keep=3
        )

        best_val_loss = np.inf
        wait = 0

        # for epoch in range(1, self.epochs + 1):
        #     epoch_train_losses = []
        #     epoch_train_parts = []

        #     # ---- Train ----
        #     for x_batch, y_batch_obs in train_dataset:
        #         with tf.GradientTape() as tape:
        #             mu_sup, mu_var, log_var_var, z = self.encoder(x_batch, training=True)
        #             reconstruction = self.decoder(z, training=True)

        #             recon_loss = mse_loss_fn(x_batch, reconstruction)
        #             kl_var = -0.5 * tf.reduce_mean(
        #                 tf.reduce_mean(
        #                     1 + log_var_var - tf.square(mu_var) - tf.exp(log_var_var), axis=1
        #                 )
        #             )
        #             supervised_loss = mse_loss_fn(y_batch_obs, mu_sup)
        #             total_loss = recon_loss + self.eta * kl_var + self.kappa * supervised_loss

        #         vars_all = self.encoder.trainable_weights + self.decoder.trainable_weights
        #         grads = tape.gradient(total_loss, vars_all)
        #         self.optimizer.apply_gradients(zip(grads, vars_all))

        #         epoch_train_losses.append(total_loss.numpy())
        #         epoch_train_parts.append([
        #             recon_loss.numpy(), kl_var.numpy(), supervised_loss.numpy()
        #         ])

        #     avg_train_loss = float(np.mean(epoch_train_losses))
        #     epoch_train_parts = np.mean(epoch_train_parts, axis=0)

        #     # ---- Val ----
        #     epoch_val_losses = []
        #     for x_batch_val, y_batch_val_obs in val_dataset:
        #         mu_sup_v, mu_var_v, log_var_v, z_v = self.encoder(x_batch_val, training=False)
        #         recon_v = self.decoder(z_v, training=False)

        #         recon_loss_v = mse_loss_fn(x_batch_val, recon_v)
        #         kl_v = -0.5 * tf.reduce_mean(
        #             tf.reduce_mean(
        #                 1 + log_var_v - tf.square(mu_var_v) - tf.exp(log_var_v), axis=1
        #             )
        #         )
        #         supervised_loss_v = mse_loss_fn(y_batch_val_obs, mu_sup_v)
        #         val_loss_batch = recon_loss_v + self.eta * kl_v + self.kappa * supervised_loss_v
        #         epoch_val_losses.append(val_loss_batch.numpy())

        #     avg_val_loss = float(np.mean(epoch_val_losses))
        #     print(f"Epoch {epoch:03d} – train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")
        #     print("train_loss_parts: [recon, KL, sup] =", epoch_train_parts)

        @tf.function()
        def train_step(x_batch, y_batch_obs):
            with tf.GradientTape() as tape:
                mu_sup, mu_var, log_var_var, z = self.encoder(x_batch, training=True)
                reconstruction = self.decoder(z, training=True)

                mu  = tf.constant(self.X_scaler.mean_.astype("float32"))
                sig = tf.constant(self.X_scaler.scale_.astype("float32"))
                reconstruction_scaled = (reconstruction - mu) / sig
        
                recon_loss = tf.cast(mse_loss_fn(x_batch, reconstruction_scaled), tf.float32)
                kl_var = tf.cast(-0.5 * tf.reduce_mean(
                    tf.reduce_mean(1 + log_var_var - tf.square(mu_var) - tf.exp(log_var_var), axis=1)
                ), tf.float32)
                supervised_loss = tf.cast(mse_loss_fn(y_batch_obs, mu_sup), tf.float32)
                total_loss = recon_loss + tf.cast(self.eta, tf.float32) * kl_var \
                             + tf.cast(self.kappa, tf.float32) * supervised_loss
        
            vars_all = self.encoder.trainable_weights + self.decoder.trainable_weights
            grads = tape.gradient(total_loss, vars_all)
            self.optimizer.apply_gradients(zip(grads, vars_all))
            return total_loss, recon_loss, kl_var, supervised_loss
        
        @tf.function()
        def val_step(x_batch_val, y_batch_val_obs):
            mu_sup_v, mu_var_v, log_var_v, z_v = self.encoder(x_batch_val, training=False)
            recon_v = self.decoder(z_v, training=False)

            mu  = tf.constant(self.X_scaler.mean_.astype("float32"))
            sig = tf.constant(self.X_scaler.scale_.astype("float32"))
            recon_v_scaled = (recon_v - mu) / sig
            
            recon_loss_v = tf.cast(mse_loss_fn(x_batch_val, recon_v_scaled), tf.float32)
            kl_v = tf.cast(-0.5 * tf.reduce_mean(
                tf.reduce_mean(1 + log_var_v - tf.square(mu_var_v) - tf.exp(log_var_v), axis=1)
            ), tf.float32)
            supervised_loss_v = tf.cast(mse_loss_fn(y_batch_val_obs, mu_sup_v), tf.float32)
            val_loss_batch = recon_loss_v + tf.cast(self.eta, tf.float32) * kl_v \
                             + tf.cast(self.kappa, tf.float32) * supervised_loss_v
            return val_loss_batch
            
        for epoch in range(1, self.epochs + 1):
            epoch_train_losses = []
            epoch_train_parts = []
        
            for x_batch, y_batch_obs in train_dataset:
                total_loss, recon_loss, kl_var, supervised_loss = train_step(x_batch, y_batch_obs)
                # Keep only lightweight scalars for logging
                epoch_train_losses.append(total_loss)
                epoch_train_parts.append([recon_loss, kl_var, supervised_loss])
        
            avg_train_loss = float(tf.reduce_mean(epoch_train_losses).numpy())
            recon_m, kl_m, sup_m = tf.reduce_mean(epoch_train_parts, axis=0)
            epoch_train_parts_mean = [float(recon_m.numpy()), float(kl_m.numpy()), float(sup_m.numpy())]
        
            # ---- Val ----
            epoch_val_losses = []
            for x_batch_val, y_batch_val_obs in val_dataset:
                val_loss_batch = val_step(x_batch_val, y_batch_val_obs)
                epoch_val_losses.append(val_loss_batch)
            avg_val_loss = float(tf.reduce_mean(epoch_val_losses).numpy())
        
            print(f"Epoch {epoch:03d} – train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")
            print("train_loss_parts: [recon, KL, sup] =", epoch_train_parts_mean)

            
            # ---- Early stopping & "best" saves ----
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                wait = 0
                # weights-only (lightweight)
                self.encoder.save_weights(os.path.join(outdir, 'best_encoder.weights.h5'))
                self.decoder.save_weights(os.path.join(outdir, 'best_decoder.weights.h5'))
                # full models (architecture + weights; easiest to reload)
                self.encoder.save(os.path.join(outdir, "encoder.keras"))
                self.decoder.save(os.path.join(outdir, "decoder.keras"))
                self.auto.save(os.path.join(outdir, "vae_eos.keras"))
                # checkpoint optimizer slots too
                self._ckpt_manager.save()
                # also keep config/scalers alongside models
                self._save_metadata(outdir)
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Stopping early at epoch {epoch}")
                    break

        # Restore best weights
        self.encoder.load_weights(os.path.join(outdir, 'best_encoder.weights.h5'))
        self.decoder.load_weights(os.path.join(outdir, 'best_decoder.weights.h5'))

        # ---- Test ----
        mse_loss_fn = tf.keras.losses.MeanSquaredError()
        test_losses = []
        for x_batch_test, y_batch_test_obs in test_dataset:
            mu_sup_t, mu_var_t, log_var_t, z_t = self.encoder(x_batch_test, training=False)
            recon_t = self.decoder(z_t, training=False)
            
            mu  = tf.constant(self.X_scaler.mean_.astype("float32"))
            sig = tf.constant(self.X_scaler.scale_.astype("float32"))
            recon_t_scaled = (recon_t - mu) / sig
            
            recon_loss_t = mse_loss_fn(x_batch_test, recon_t_scaled)
            kl_var_t = -0.5 * tf.reduce_mean(
                tf.reduce_mean(1 + log_var_t - tf.square(mu_var_t) - tf.exp(log_var_t), axis=1)
            )
            supervised_loss_t = mse_loss_fn(y_batch_test_obs, mu_sup_t)
            test_loss_batch = recon_loss_t + self.eta * kl_var_t + self.kappa * supervised_loss_t
            test_losses.append(test_loss_batch.numpy())

        avg_test_loss = float(np.mean(test_losses))
        print(f"Test Loss: {avg_test_loss:.6f}")

        # final metadata save (ensures scalers/config exist even if best never improved after first epoch)
        self._save_metadata(outdir)
        return avg_test_loss

    # -------- Saving / Loading --------
    def _save_metadata(self, outdir):
        """Save scalers + config JSON."""
        os.makedirs(outdir, exist_ok=True)
        if self.X_scaler is not None:
            joblib.dump(self.X_scaler, os.path.join(outdir, "X_scaler.joblib"))
        if self.y_scaler is not None:
            joblib.dump(self.y_scaler, os.path.join(outdir, "y_scaler.joblib"))
        cfg = {
            "input_dim": self.input_dim,
            "latent_dim_supervised": self.latent_dim_supervised,
            "latent_dim_variational": self.latent_dim_variational,
            "eta": self.eta, "kappa": self.kappa,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs, "patience": self.patience,
        }
        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    def save_all(self, outdir="artifacts"):
        """Optional explicit save (full models + ckpt + metadata)."""
        os.makedirs(outdir, exist_ok=True)
        if self.encoder is not None:
            self.encoder.save(os.path.join(outdir, "encoder.keras"))
            self.encoder.save_weights(os.path.join(outdir, "encoder.weights.h5"))
        if self.decoder is not None:
            self.decoder.save(os.path.join(outdir, "decoder.keras"))
            self.decoder.save_weights(os.path.join(outdir, "decoder.weights.h5"))
        if self.auto is not None:
            self.auto.save(os.path.join(outdir, "vae_eos.keras"))
            self.auto.save_weights(os.path.join(outdir, "vae_eos.weights.h5"))
        # Save a checkpoint with optimizer slots
        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate or 1e-4)
        ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder, optimizer=self.optimizer)
        manager = tf.train.CheckpointManager(ckpt, os.path.join(outdir, "ckpts"), max_to_keep=3)
        manager.save()
        # Metadata
        self._save_metadata(outdir)

    @staticmethod
    def load_all(outdir="artifacts"):
        """
        Load saved models & metadata.
        Returns: (encoder, decoder, auto, X_scaler, y_scaler)
        """
        encoder = tf.keras.models.load_model(os.path.join(outdir, "encoder.keras"))
        decoder = tf.keras.models.load_model(os.path.join(outdir, "decoder.keras"))
        auto    = tf.keras.models.load_model(os.path.join(outdir, "vae_eos.keras"))
        X_scaler = joblib.load(os.path.join(outdir, "X_scaler.joblib"))
        y_scaler = joblib.load(os.path.join(outdir, "y_scaler.joblib")) if os.path.exists(
            os.path.join(outdir, "y_scaler.joblib")
        ) else None
        # Optionally restore optimizer slots
        optimizer = tf.keras.optimizers.Adam()
        ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
        latest = tf.train.latest_checkpoint(os.path.join(outdir, "ckpts"))
        if latest:
            ckpt.restore(latest).expect_partial()
        return encoder, decoder, auto, X_scaler, y_scaler
