package student;

public class HuffmanNode {

    // Private variables for the Huffman Node
    private HuffmanNode zero; // Child node for '0'
    private HuffmanNode one;  // Child node for '1'
    private Character data;   // Data stored at this node (null if internal node)

    // Constructor for an internal node
    public HuffmanNode(HuffmanNode zero, HuffmanNode one) {
        this.zero = zero;
        this.one = one;
        this.data = null; // Internal node has no data
    }

    // Constructor for a leaf node
    public HuffmanNode(char data) {
        this.data = data;
        this.zero = null;
        this.one = null; // Leaf nodes have no children
    }

    // Getter and Setter for the 'zero' child
    public HuffmanNode getZero() {
        return zero;
    }

    public void setZero(HuffmanNode zero) {
        this.zero = zero;
    }

    // Getter and Setter for the 'one' child
    public HuffmanNode getOne() {
        return one;
    }

    public void setOne(HuffmanNode one) {
        this.one = one;
    }

    // Getter and Setter for the 'data' field
    public Character getData() {
        return data;
    }

    public void setData(Character data) {
        this.data = data;
    }

    // Checks if the node is a leaf (has no children)
    public boolean isLeaf() {
        return zero == null && one == null && data != null;
    }

    // Checks if the node is valid for a Huffman code tree
    public boolean isValidNode() {
        if (isLeaf()) {
            return true; // Leaf node is valid
        }
        // Internal node must have both children and no data
        return zero != null && one != null && data == null;
    }

    // Recursively checks if the node and all its descendants are valid
    public boolean isValidTree() {
        // Check if the current node is valid
        if (!isValidNode()) {
            return false;
        }
        // Recursively validate the children (if present)
        boolean validZero = zero == null || zero.isValidTree();
        boolean validOne = one == null || one.isValidTree();
        return validZero && validOne;
    }
}

