from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
import torch
import torch.distributed as dist
import os

# Training script
def train(device, model, train_loader, val_loader, criterion, optimizer,scheduler,writer,model_path,args,fold):
    model = model.to(device)
    best_accuracy = 0.0
    accuracy = Accuracy(task=args.task, num_classes=args.num_classes).to(device)
    precision = Precision(task=args.task, average='macro', num_classes=args.num_classes).to(device)
    recall = Recall(task=args.task,, average='macro', num_classes=args.num_classes).to(device)
    auroc = AUROC(task=args.task,).to(device)
    f1 = F1Score(num_classes=args.num_classes, task=args.task,, average='macro').to(device)

    train_iter_idx = 0
    sum_loss = 0
    record_loss = []
    freq = 50
    
    global_loss=0
    for epoch in range(args.num_epochs):
        model.train()
        global_loss = 0.0
        train_loader.sampler.set_epoch(epoch) # DDP setting
        for images, coords, labels, _, _ ,distances_matrix in train_loader:

            train_iter_idx += 1

            images =images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images,coords,distances_matrix)
            loss = criterion(outputs, labels)
            loss.backward() 

            loss_comm = loss.detach().clone() 
            torch.distributed.all_reduce(loss_comm, op = torch.distributed.ReduceOp.SUM)
            global_avg_loss = loss_comm / dist.get_world_size() 

            sum_loss += global_avg_loss
            global_loss += global_avg_loss
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            accuracy(predicted, labels.data)
            precision(predicted, labels.data)
            recall(predicted, labels.data)
            f1(predicted, labels.data)
            auroc(predicted, labels.data)

            if  train_iter_idx % freq == 0:
                sum_loss = sum_loss / freq
                record_loss.append(sum_loss)
                print(f"batch_num {train_iter_idx}/{len(train_loader)}:Epoch:{epoch + 1} Train Loss: {sum_loss:.4f}")
                sum_loss = 0

        global_loss/=len(train_loader)
        scheduler.step()
        acc = accuracy.compute() 
        prec = precision.compute() 
        rec = recall.compute() 
        f1_score = f1.compute()
        auroc_score = auroc.compute()
        writer.add_scalar('Train/Loss', global_loss, epoch)
        writer.add_scalar('Train/Accuracy', acc, epoch)
        writer.add_scalar('Train/Precision', prec, epoch)
        writer.add_scalar('Train/Recall', rec, epoch)
        writer.add_scalar('Train/AUROC', auroc_score, epoch)
        writer.add_scalar('Train/F1', f1_score, epoch)
        
        
        print(f"Epoch {epoch + 1}/{args.num_epochs}:  Train Loss: {global_loss:.4f}, Train Accuracy: {acc:.4f}, Train precision: {prec:.4f}, Train recall: {rec:.4f}, Train f1: {f1_score:.4f}, Train auroc: {auroc_score:4f}")
        # Verify once every validate_interval
        if (epoch + 1) % args.val_check_interval == 0:
            loss, acc = validate(model, val_loader, criterion, device, epoch, writer)
            if acc > best_accuracy:
                best_accuracy = acc
                # DDP setting
                if dist.get_rank() == 0:
                    filename=f'model_fold{fold}.pth'
                    torch.save(model.state_dict(), os.path.join(model_path,filename))
                print(f"Best acc changed, acc is {acc:.4f}, loss is {loss:.4f}. Model saved.")
            else:
                print(f"Best acc not changed, acc is {acc:.4f}, loss is {loss:.4f}. Model not saved.")
# Validate script
def validate(model, val_loader, criterion, device, epoch, writer):
    val_loader.sampler.set_epoch(epoch)
    model.eval()
    val_loss = 0.0

    val_accuracy = Accuracy(task=args.task, num_classes=args.num_classes).to(device)
    val_precision = Precision(task=args.task, average='macro', num_classes=args.num_classes).to(device)
    val_recall = Recall(task=args.task, average='macro', num_classes=args.num_classes).to(device)
    val_auroc = AUROC(task=args.task).to(device)
    val_f1 = F1Score(num_classes=args.num_classes, task=args.task, average='macro').to(device)
    with torch.no_grad():
        for images, coords, labels, _, _ ,distances_matrix in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images,coords,distances_matrix)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            
            loss_comm = loss.detach().clone() 
            torch.distributed.all_reduce(loss_comm, op = torch.distributed.ReduceOp.SUM)
            avg_loss = loss_comm / dist.get_world_size() 
            val_loss += avg_loss

            val_accuracy(predicted, labels.data)
            val_precision(predicted, labels.data)
            val_recall(predicted, labels.data)
            val_f1(predicted, labels.data)
            val_auroc(predicted, labels.data)
    average_loss = val_loss / len(val_loader)
    acc = val_accuracy.compute() 
    prec = val_precision.compute() 
    rec = val_recall.compute() 
    f1_score = val_f1.compute()
    auroc_score = val_auroc.compute()
    writer.add_scalar('Val/Loss', average_loss, epoch)
    writer.add_scalar('Val/Accuracy', acc, epoch)
    writer.add_scalar('Val/Precision', prec, epoch)
    writer.add_scalar('Val/Recall', rec, epoch)
    writer.add_scalar('Val/AUROC', auroc_score, epoch)
    writer.add_scalar('Val/F1', f1_score, epoch)
    print(f"Validation Loss: {average_loss:.6f}, Validation Accuracy: {acc:.6f}, Validation precision: {prec:.6f}, Validation recall: {rec:.6f}, Validation f1: {f1_score:.6f}, Validation auroc: {auroc_score:6f}")
    val_accuracy.reset()
    val_precision.reset()
    val_recall.reset()
    val_f1.reset()
    val_auroc.reset()
    
    return average_loss, acc
