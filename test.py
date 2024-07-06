import torch
import torch.nn.functional as F


def test(model, device, test_loader):
    model.eval()
    final_loss = 0.
    correct = 0
    batch_num = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, 0.)
            loss = F.cross_entropy(output, target)
            '''
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            '''
            top5_pred = output.topk(5, dim=1).indices
            correct += sum([1 if target[i] in top5_pred[i] else 0 for i in range(len(target))])
            
            final_loss += loss.item()
            batch_num += 1
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(final_loss / batch_num, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

