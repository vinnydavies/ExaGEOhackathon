"""Forecasting models for a univariate time series (e.g. temperature).

Both models consume the same windowed input shape produced by
`make_supervised_sequences`: `(1, lookback, samples)` (features, timesteps, batch).
"""

using Flux
using Random: randperm

struct LSTMForecaster
    lstm
    head
end
Flux.@layer LSTMForecaster

"""Single-layer LSTM regressor predicting `horizon` future steps."""
function LSTMForecaster(; units::Int = 32, horizon::Int = 1)
    return LSTMForecaster(Flux.LSTM(1 => units), Dense(units => horizon))
end

function (m::LSTMForecaster)(x)
    Flux.reset!(m.lstm)
    hidden = m.lstm(x)
    return m.head(hidden[:, end, :])
end

struct DenseForecaster
    net
end
Flux.@layer DenseForecaster

"""Classic feed-forward (MLP) regressor over a flattened lookback window."""
function DenseForecaster(; lookback::Int, units::Int = 32, horizon::Int = 1)
    net = Chain(
        x -> reshape(x, lookback, :),
        Dense(lookback => units, relu),
        Dense(units => units, relu),
        Dense(units => horizon),
    )
    return DenseForecaster(net)
end

(m::DenseForecaster)(x) = m.net(x)

# index into the last (sample) dimension of an array of any rank
take_obs(a::AbstractArray, idx) = a[ntuple(_ -> Colon(), ndims(a) - 1)..., idx]

"""Train any forecaster (LSTM, Dense, etc.) with early stopping on validation loss.

Returns a history dict with `loss` and `val_loss` vectors, one entry per epoch.
"""
function train_forecaster(
    model,
    X_train::AbstractArray,
    y_train::AbstractArray;
    validation_split::Float64 = 0.2,
    epochs::Int = 50,
    batch_size::Int = 32,
    learning_rate::Float64 = 1e-3,
    patience::Int = 5,
)
    n_samples = size(X_train)[end]
    n_val = floor(Int, n_samples * validation_split)
    n_train = n_samples - n_val
    perm = randperm(n_samples)
    train_idx, val_idx = perm[1:n_train], perm[(n_train + 1):end]

    X_tr, y_tr = take_obs(X_train, train_idx), take_obs(y_train, train_idx)
    X_val, y_val = take_obs(X_train, val_idx), take_obs(y_train, val_idx)

    train_loader = Flux.DataLoader((X_tr, y_tr); batchsize = batch_size, shuffle = true)
    opt_state = Flux.setup(Flux.Adam(learning_rate), model)

    history = Dict("loss" => Float64[], "val_loss" => Float64[])
    best_val_loss = Inf
    best_state = nothing
    epochs_without_improvement = 0

    for _ in 1:epochs
        train_loss_sum = 0.0
        n_seen = 0
        for (x_batch, y_batch) in train_loader
            loss, grads = Flux.withgradient(model) do m
                Flux.mse(m(x_batch), y_batch)
            end
            Flux.update!(opt_state, model, grads[1])
            train_loss_sum += loss * size(x_batch)[end]
            n_seen += size(x_batch)[end]
        end
        train_loss = train_loss_sum / n_seen
        val_loss = n_val > 0 ? Float64(Flux.mse(model(X_val), y_val)) : 0.0

        push!(history["loss"], train_loss)
        push!(history["val_loss"], val_loss)

        if val_loss < best_val_loss
            best_val_loss = val_loss
            best_state = deepcopy(Flux.state(model))
            epochs_without_improvement = 0
        else
            epochs_without_improvement += 1
            epochs_without_improvement >= patience && break
        end
    end

    best_state !== nothing && Flux.loadmodel!(model, best_state)
    return history
end
