import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import time


def binary_accuracy(preds, y):
    '''Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    input:
    preds: predicted values
    y: truth batch values

    output: the percent of predictions that were correct in the batch'''

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    '''training function. Does a single iteration over our training data bucket iterator object and performs backpropagation on our model
    input:
    model: our BERTGRUSentiment model
    iterator: the training bucket iterator
    optimizer: our optimizer (IE: Adam, vanilla SGD)
    criterion: our loss function

    output: the average training loss and average training accuracy
    note that the model is updated during this process
    '''
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    '''Evaluates our model on a new dataset for a single iteration, making sure not to call backpropagation.
    input:
    model: our BERTGRUSentiment model
    iterator: the bucket iterator of choice, either validation or testing
    criterion: our loss function

    output: the average validation (or testing) loss and average validation (or testing) accuracy
    '''
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            precision = precision_score(predictions.cpu().data.numpy() > 0.5, batch.label.cpu().data.numpy())
            recall = recall_score(predictions.cpu().data.numpy() > 0.5, batch.label.cpu().data.numpy())
            f1 = f1_score(predictions.cpu().data.numpy() > 0.5, batch.label.cpu().data.numpy())

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator), precision, recall, f1


def epoch_time(start_time, end_time):
    '''helper function to track how long training process takes'''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def fit(n_epochs, model, train_iter, valid_iter, optimizer, criterion, model_name: str):
    '''uses train/evaluate in all one routine

    TODO:
    1) change gradient l2 norm printout to file write out.
    2) write out all metrics out to csv
    3) write out best model to .pt file
    4) end of training move files in 2/3 to S3
    '''

    best_valid_loss = float('inf')

    for epoch in tqdm(range(n_epochs)):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iter, optimizer, criterion)
        with open(f'{model_name} gradients.txt', "a") as f:
            f.write(f'epoch # {epoch + 1}')
            for p in list(filter(lambda p: p.grad is not None, model.parameters())):
                f.write(str(p.grad.data.norm(2).item()))
            f.write('\n*********\n')
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_iter, criterion)
        print(f'validation done epoch # {epoch + 1}')

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # only save the model if the
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_name + '.pt')
        with open(f'{model_name} results.txt', "a") as f:
            f.write(f'Epoch: {epoch + 1} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            f.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            f.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% |  Val. Precision: '
                    f'{valid_precision * 100:.2f}|  Val. Recall: {valid_recall * 100:.2f} |  '
                    f'Val. F1: {valid_f1 * 100:.2f}')
