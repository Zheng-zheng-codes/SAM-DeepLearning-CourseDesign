import torch


def train_one_epoch(model, train_loader, criterion, optimizer, device, optimizer_name):
    """
    Train model for one epoch.

    Args:
        model: neural network model
        train_loader: training dataloader
        criterion: loss function
        optimizer: optimizer
        device: cuda or cpu
        optimizer_name: sgd, adam, or sam

    Returns:
        train_loss, train_acc
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        if optimizer_name == "sam":
            # First forward-backward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.second_step(zero_grad=True)

        else:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Record training metrics using current model
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)

            total_loss += loss.item() * images.size(0)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    return train_loss, train_acc


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test set.

    Returns:
        test_loss, test_acc
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)

            total_loss += loss.item() * images.size(0)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    test_loss = total_loss / total
    test_acc = correct / total

    return test_loss, test_acc