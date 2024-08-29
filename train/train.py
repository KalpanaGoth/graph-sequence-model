def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=100, patience=10):
    """
    Train the graph-based model.
    """
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Assume each batch contains nodes, edges, and labels
            nodes, edges, labels = batch

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(nodes, edges)  # Pass both nodes and edges

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Step the scheduler
        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")
