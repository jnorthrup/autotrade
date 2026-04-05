with open('model.py', 'r') as f:
    text = f.read()

old = "self.x_pixels, self.hidden_size, self.num_nodes, self.num_heads = x_pixels, max(1, int(num_nodes)), hidden_size, num_heads"
new = "self.x_pixels, self.hidden_size, self.num_nodes, self.num_heads = x_pixels, hidden_size, max(1, int(num_nodes)), num_heads"

if old in text:
    text = text.replace(old, new)
    with open('model.py', 'w') as f:
        f.write(text)
    print("Reverted model.py back to correct assignment")
