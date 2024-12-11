package student;

import provided.BinarySequence;

public class HuffmanCodeTree {
    // Root node of the Huffman code tree
    private final HuffmanNode root;

    // Constructor: Creates a Huffman code tree with a given root node
    public HuffmanCodeTree(HuffmanNode root) {
        this.root = root;
    }

    // Constructor: Builds a Huffman code tree from a HuffmanCodeBook
    public HuffmanCodeTree(HuffmanCodeBook codebook) {
        this.root = new HuffmanNode(null, null); // Create an initially invalid root node
        for (char c : codebook.getAllCharacters()) {
            BinarySequence sequence = codebook.getSequence(c);
            put(sequence, c); // Add each character and its sequence to the tree
        }
    }

    // Validates if the tree rooted at this node is a valid Huffman code tree
    public boolean isValid() {
        return root.isValidTree();
    }

    // Inserts a character into the tree using a given binary sequence
    public void put(BinarySequence seq, char letter) {
        HuffmanNode current = root;

        // Traverse the sequence bit by bit
        for (boolean bit : seq) {
            if (bit) { // If true (1), follow or create the "one" child
                if (current.getOne() == null) {
                    current.setOne(new HuffmanNode(null, null));
                }
                current = current.getOne();
            } else { // If false (0), follow or create the "zero" child
                if (current.getZero() == null) {
                    current.setZero(new HuffmanNode(null, null));
                }
                current = current.getZero();
            }
        }

        // Assign the character to the leaf node
        current.setData(letter);
    }

    // Decodes a binary sequence into a string
    public String decode(BinarySequence s) {
        StringBuilder result = new StringBuilder();
        HuffmanNode current = root;

        // Traverse the sequence bit by bit
        for (boolean bit : s) {
            if (bit) {
                current = current.getOne();
            } else {
                current = current.getZero();
            }

            // If a leaf node is reached, append the data and reset to root
            if (current.isLeaf()) {
                result.append(current.getData());
                current = root;
            }
        }
        return result.toString();
    }
}

