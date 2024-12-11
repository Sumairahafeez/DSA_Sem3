package student;

import provided.BinarySequence;

// you may implement any interface you want here
import java.util.HashMap;
import java.util.Map;

public class HuffmanCodeBook {

    // Map to store characters and their corresponding binary sequences
    private final Map<Character, BinarySequence> codeBook;

    // Constructor: initializes an empty codebook
    public HuffmanCodeBook() {
        codeBook = new HashMap<>();
    }

    // Adds a character and its binary sequence to the codebook
    public void addSequence(char c, BinarySequence seq) {
        codeBook.put(c, seq);
    }

    // Checks if a given character exists in the codebook
    public boolean contains(char letter) {
        return codeBook.containsKey(letter);
    }

    // Checks if the codebook contains all characters in a given string
    public boolean containsAll(String letters) {
        for (char letter : letters.toCharArray()) {
            if (!contains(letter)) {
                return false;
            }
        }
        return true;
    }

    // Retrieves the binary sequence associated with a character
    public BinarySequence getSequence(char c) {
        return codeBook.getOrDefault(c, null);
    }

    // Encodes a string into a binary sequence
    public BinarySequence encode(String s) {
        BinarySequence result = new BinarySequence();
        for (char c : s.toCharArray()) {
            BinarySequence seq = getSequence(c);
            if (seq != null) {
                result.append(seq);
            }
        }
        return result;
    }

    // Allows iteration over all characters in the codebook
    public Iterable<Character> getAllCharacters() {
        return codeBook.keySet();
    }
}
