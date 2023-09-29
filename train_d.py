import mindspore
from mindspore import nn
from utils import load_config, Meter, cal_score
from can import CAN
from dataset import get_crohme_dataset


params = load_config('/cache/code/can/config.yaml')
device = params['device']
mindspore.set_context(device_target=device)
mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

train_loader, eval_loader = get_crohme_dataset(params)
model = CAN(params)
optimizer = getattr(mindspore.nn, params['optimizer'])(model.trainable_params(), learning_rate=float(params['lr']), weight_decay=float(params['weight_decay']))
init_epoch = 0

# Get gradient function
grad_fn = mindspore.ops.value_and_grad(model, None, optimizer.parameters, has_aux=True)


# Define function of one-step training
def train_step(images, image_masks, labels, label_masks):
    (loss, probs, counting_preds, word_loss, counting_loss), grads = grad_fn(images, image_masks, labels, label_masks)
    if params['gradient_clip']:
        grads = mindspore.ops.clip_by_global_norm(grads)
    loss = mindspore.ops.depend(loss, optimizer(grads))
    return loss


def train_loop(dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, batch_data in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(*batch_data)
        if batch % 10 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss}  [{current:>3d}/{size:>3d}]")


def eval_loop(dataset):
    model.set_train(False)
    loss_meter = Meter()
    word_right, exp_right, length, cal_num = 0, 0, 0, 0
    for batch_idx, (images, image_masks, labels, label_masks) in enumerate(dataset.create_tuple_iterator()):
        batch, time = labels.shape[:2]
        loss, probs, counting_preds, word_loss, counting_loss = model(images, image_masks, labels, label_masks,
                                                                      is_train=False)
        loss_meter.add(loss.item())

        wordRate, ExpRate = cal_score(probs, labels, label_masks)
        word_right = word_right + wordRate * time
        exp_right = exp_right + ExpRate * batch
        length = length + time
        cal_num = cal_num + batch
    print('loss:{}'.format(loss_meter.mean),
          'word_right / length:{}'.format(word_right / length),
          'exp_right / cal_num:'.format(exp_right / cal_num))


print('Start training...')
for epoch in range(init_epoch, params['epochs']):
    print('-----epoch: ', epoch, 'Started.')
    if epoch % 20 == 0:
        mindspore.save_checkpoint(model, "/cache/output/model-"+str(epoch)+".ckpt")
    # train_loop(train_loader)
    eval_loop(eval_loader)
    print('-----epoch: ', epoch, 'Finished.')
print('Training Success!')
mindspore.save_checkpoint(model, "/cache/output/model-final.ckpt")