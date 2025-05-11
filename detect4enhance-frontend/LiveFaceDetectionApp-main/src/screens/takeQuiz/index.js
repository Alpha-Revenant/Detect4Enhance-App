import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, FlatList } from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';
import { ScreenWrapper } from 'react-native-screen-wrapper';
import { AppColors } from '../../utils/AppCollors';
import CustomHeader from '../../components/header';
import { height, width } from '../../methord/Dimentions';

// Sample MCQs
const programmingMCQs = [
    {
        question: "Which data type is used to create a variable that should store text?",
        options: ["String", "Integer", "Float", "Boolean"],
        correctAnswer: "String"
    },
    {
        question: "What is the correct syntax to output 'Hello World' in Python?",
        options: ["echo 'Hello World'", "print('Hello World')", "console.log('Hello World')", "System.out.println('Hello World')"],
        correctAnswer: "print('Hello World')"
    },
    {
        question: "Which operator is used to assign a value to a variable in most programming languages?",
        options: ["=", "==", ":", "::"],
        correctAnswer: "="
    },
    {
        question: "What is the output of 3 + 2 * 2 in most programming languages?",
        options: ["10", "7", "9", "5"],
        correctAnswer: "7"
    },
    {
        question: "What is the term for a block of code that performs a specific task in a program?",
        options: ["Loop", "Function", "Variable", "Array"],
        correctAnswer: "Function"
    },
    {
        question: "What does HTML stand for?",
        options: [
            "Hyper Text Markup Language",
            "Hyperlinks and Text Markup Language",
            "Home Tool Markup Language",
            "High Text Markup Language"
        ],
        correctAnswer: "Hyper Text Markup Language"
    },
    {
        question: "Which keyword is used to declare a constant variable in JavaScript?",
        options: ["let", "var", "const", "static"],
        correctAnswer: "const"
    },
    {
        question: "What does the 'break' statement do in a loop?",
        options: [
            "Terminates the loop",
            "Skips the current iteration",
            "Pauses the loop temporarily",
            "None of the above"
        ],
        correctAnswer: "Terminates the loop"
    },
    {
        question: "Which symbol is used for comments in Python?",
        options: ["//", "#", "/*", "--"],
        correctAnswer: "#"
    },
    {
        question: "Which language is primarily used for developing iOS applications?",
        options: ["Swift", "Java", "Python", "Ruby"],
        correctAnswer: "Swift"
    },
    {
        question: "What is the time complexity of a binary search algorithm?",
        options: ["O(n)", "O(log n)", "O(n^2)", "O(1)"],
        correctAnswer: "O(log n)"
    },
    {
        question: "Which of these is NOT a valid JavaScript data type?",
        options: ["Number", "Boolean", "Object", "Character"],
        correctAnswer: "Character"
    },
    {
        question: "Which loop executes at least once regardless of the condition?",
        options: ["for loop", "while loop", "do-while loop", "foreach loop"],
        correctAnswer: "do-while loop"
    },
    {
        question: "Which of the following is an example of a relational operator?",
        options: ["+", "=", "<=", "*"],
        correctAnswer: "<="
    },
    {
        question: "What is the purpose of a constructor in object-oriented programming?",
        options: [
            "To allocate memory",
            "To initialize an object",
            "To define methods",
            "To destroy an object"
        ],
        correctAnswer: "To initialize an object"
    }
];

const databaseMCQs = [
    {
        question: "What does SQL stand for?",
        options: [
            "Structured Query Language",
            "System Query Language",
            "Simple Query Language",
            "Sequential Query Language"
        ],
        correctAnswer: "Structured Query Language"
    },
    {
        question: "Which of the following is a primary key?",
        options: [
            "A unique identifier for each record",
            "A field that allows duplicate values",
            "A field with no data",
            "A foreign key"
        ],
        correctAnswer: "A unique identifier for each record"
    },
    {
        question: "What does the acronym ACID stand for in databases?",
        options: [
            "Atomicity, Consistency, Isolation, Durability",
            "Accuracy, Consistency, Integrity, Durability",
            "Atomicity, Concurrency, Isolation, Dependency",
            "Accuracy, Concurrency, Isolation, Dependency"
        ],
        correctAnswer: "Atomicity, Consistency, Isolation, Durability"
    },
    {
        question: "Which SQL command is used to retrieve data from a database?",
        options: ["SELECT", "UPDATE", "DELETE", "INSERT"],
        correctAnswer: "SELECT"
    },
    {
        question: "What is a foreign key in a database?",
        options: [
            "A unique identifier within a table",
            "A field in a table that links to the primary key of another table",
            "A key used for encryption",
            "A placeholder for missing data"
        ],
        correctAnswer: "A field in a table that links to the primary key of another table"
    },
    {
        question: "Which of these is a NoSQL database?",
        options: ["MySQL", "MongoDB", "PostgreSQL", "Oracle"],
        correctAnswer: "MongoDB"
    },
    {
        question: "What is the purpose of an index in a database?",
        options: [
            "To store data in alphabetical order",
            "To enforce data integrity",
            "To improve query performance",
            "To create relationships between tables"
        ],
        correctAnswer: "To improve query performance"
    },
    {
        question: "Which normalization form eliminates repeating groups?",
        options: ["1NF", "2NF", "3NF", "BCNF"],
        correctAnswer: "1NF"
    },
    {
        question: "What is the result of a SQL JOIN operation?",
        options: [
            "Combines columns from multiple tables",
            "Deletes rows from a table",
            "Adds new rows to a table",
            "Duplicates a table structure"
        ],
        correctAnswer: "Combines columns from multiple tables"
    },
    {
        question: "What is a composite key?",
        options: [
            "A single field that uniquely identifies a record",
            "A combination of two or more fields that uniquely identifies a record",
            "A key that can accept NULL values",
            "A foreign key referencing multiple tables"
        ],
        correctAnswer: "A combination of two or more fields that uniquely identifies a record"
    },
    {
        question: "Which SQL statement is used to create a new table?",
        options: ["CREATE TABLE", "INSERT INTO", "ALTER TABLE", "DROP TABLE"],
        correctAnswer: "CREATE TABLE"
    },
    {
        question: "What is a database transaction?",
        options: [
            "A single unit of work that must be completed entirely or not at all",
            "A command used to retrieve data",
            "An operation to back up a database",
            "A relationship between two tables"
        ],
        correctAnswer: "A single unit of work that must be completed entirely or not at all"
    },
    {
        question: "Which of the following is an example of a DDL command?",
        options: ["CREATE", "SELECT", "INSERT", "UPDATE"],
        correctAnswer: "CREATE"
    },
    {
        question: "What is a trigger in a database?",
        options: [
            "A scheduled task to update data",
            "A special stored procedure executed when certain events occur",
            "A key used to lock rows",
            "A type of index for faster search"
        ],
        correctAnswer: "A special stored procedure executed when certain events occur"
    },
    {
        question: "Which type of database backup includes all changes since the last full backup?",
        options: ["Full backup", "Incremental backup", "Differential backup", "Snapshot"],
        correctAnswer: "Differential backup"
    }
];
const dataStructureMCQs = [
    {
        question: "Which data structure uses LIFO (Last In, First Out) principle?",
        options: ["Queue", "Stack", "Array", "Heap"],
        correctAnswer: "Stack"
    },
    {
        question: "What is the time complexity of searching for an element in a balanced binary search tree (BST)?",
        options: ["O(log n)", "O(n)", "O(n^2)", "O(1)"],
        correctAnswer: "O(log n)"
    },
    {
        question: "Which data structure is used in Breadth-First Search (BFS) of a graph?",
        options: ["Stack", "Queue", "Array", "Linked List"],
        correctAnswer: "Queue"
    },
    {
        question: "What is the best-case time complexity of the Quick Sort algorithm?",
        options: ["O(n)", "O(n log n)", "O(n^2)", "O(log n)"],
        correctAnswer: "O(n log n)"
    },
    {
        question: "Which of the following is NOT a linear data structure?",
        options: ["Array", "Stack", "Queue", "Graph"],
        correctAnswer: "Graph"
    },
    {
        question: "Which of the following is the correct definition of a binary tree?",
        options: [
            "A tree where every node has at most two children",
            "A tree where every node has at least two children",
            "A tree where nodes are arranged in a sorted order",
            "A tree where every level has the maximum number of nodes"
        ],
        correctAnswer: "A tree where every node has at most two children"
    },
    {
        question: "Which operation is the most time-consuming in a Linked List?",
        options: ["Insertion", "Deletion", "Traversal", "Search"],
        correctAnswer: "Search"
    },
    {
        question: "What is the time complexity of inserting an element at the beginning of a singly linked list?",
        options: ["O(1)", "O(n)", "O(log n)", "O(n^2)"],
        correctAnswer: "O(1)"
    },
    {
        question: "Which data structure is typically used for implementing recursion?",
        options: ["Queue", "Stack", "Heap", "Graph"],
        correctAnswer: "Stack"
    },
    {
        question: "Which traversal method is used to obtain a sorted sequence from a binary search tree?",
        options: ["Inorder", "Preorder", "Postorder", "Level order"],
        correctAnswer: "Inorder"
    },
    {
        question: "What is a circular queue?",
        options: [
            "A queue implemented using a circular linked list",
            "A queue where the last position is connected back to the first position",
            "A queue that allows insertion at both ends",
            "A queue that can store only circular data"
        ],
        correctAnswer: "A queue where the last position is connected back to the first position"
    },
    {
        question: "Which hashing technique resolves collisions using a linked list?",
        options: ["Open Addressing", "Chaining", "Linear Probing", "Double Hashing"],
        correctAnswer: "Chaining"
    },
    {
        question: "What is a priority queue?",
        options: [
            "A queue where elements are removed in priority order",
            "A queue where elements are sorted",
            "A queue with a fixed size",
            "A queue that uses LIFO principle"
        ],
        correctAnswer: "A queue where elements are removed in priority order"
    },
    {
        question: "What is the height of a complete binary tree with n nodes?",
        options: ["log2(n)", "n", "n/2", "n-1"],
        correctAnswer: "log2(n)"
    },
    {
        question: "What is the maximum number of edges in a simple graph with n vertices?",
        options: ["n(n-1)/2", "n^2", "n(n+1)/2", "n-1"],
        correctAnswer: "n(n-1)/2"
    }
];


export default function TakeQuiz() {
    const navigation = useNavigation();
    const routes = useRoute()

    const quizOf = routes.params?.QuizOf
    console.log('====================================');
    console.log("quiz", quizOf);
    console.log('====================================');
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
    const [selectedAnswer, setSelectedAnswer] = useState(null);
    const [score, setScore] = useState(0);
    const [isQuizFinished, setIsQuizFinished] = useState(false);
    const [subjectOfQuiz, setSubjectOfQuiz] = useState([])


    useEffect(() => {
        if (quizOf === 'Programming fundamental') {
            setSubjectOfQuiz(programmingMCQs)
        } else if (quizOf === 'Database') {
            setSubjectOfQuiz(databaseMCQs)
        } else if (quizOf === 'Data Structures') {
            setSubjectOfQuiz(dataStructureMCQs)
        }
    }, [quizOf])

    const currentQuestion = subjectOfQuiz[currentQuestionIndex] ? subjectOfQuiz[currentQuestionIndex] : [];

    // Handle answer selection
    const handleAnswerSelect = (option) => {
        setSelectedAnswer(option);
    };

    // Handle next question or finish quiz
    const handleNextQuestion = () => {
        if (selectedAnswer === currentQuestion.correctAnswer) {
            setScore(score + 1);
        }
        setSelectedAnswer(null);

        if (currentQuestionIndex < subjectOfQuiz.length - 1) {
            setCurrentQuestionIndex(currentQuestionIndex + 1);
        } else {
            setIsQuizFinished(true);
        }
    };

    // Reset the quiz
    const resetQuiz = () => {
        setCurrentQuestionIndex(0);
        setScore(0);
        setIsQuizFinished(false);
    };

    return (
        <ScreenWrapper statusBarColor={AppColors.primary}>
            <View style={styles.container}>
                <CustomHeader navigation={navigation} />



                {/* Quiz Content */}
                {!isQuizFinished ? (

                    <View style={styles.quizContent}>
                        <View style={styles.headerView}>
                            <Text style={styles.headerText}>Take Quiz</Text>
                        </View>
                        {/* Question */}
                        <Text style={styles.questionText}>{currentQuestion.question}</Text>

                        {/* Options */}
                        <FlatList
                            data={currentQuestion.options}
                            keyExtractor={(item) => item}
                            renderItem={({ item }) => (
                                <TouchableOpacity
                                    style={[
                                        styles.optionButton,
                                        {
                                            backgroundColor: selectedAnswer === item ? AppColors.primary : AppColors.secondary,
                                        },
                                    ]}
                                    onPress={() => handleAnswerSelect(item)}
                                >
                                    <Text style={styles.optionText}>{item}</Text>
                                </TouchableOpacity>
                            )}
                        />

                        {/* Submit Button */}
                        <TouchableOpacity
                            style={styles.submitButton}
                            onPress={handleNextQuestion}
                            disabled={!selectedAnswer}
                        >
                            <Text style={styles.submitButtonText}>
                                {currentQuestionIndex < subjectOfQuiz.length - 1 ? 'Next' : 'Finish'}
                            </Text>
                        </TouchableOpacity>
                    </View>
                ) : (
                    <View style={styles.resultView}>
                        <Text style={styles.resultText}>Quiz Completed!</Text>
                        <Text style={styles.scoreText}>Your Score: {score} / {subjectOfQuiz.length}</Text>

                        {/* Restart Button */}
                        <TouchableOpacity style={styles.restartButton} onPress={resetQuiz}>
                            <Text style={styles.restartButtonText}>Restart Quiz</Text>
                        </TouchableOpacity>
                    </View>
                )}
            </View>
        </ScreenWrapper>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: AppColors.primeryLight,
    },
    headerView: {
        alignSelf: 'center',
        marginTop: height(3),
        marginBottom: height(8),
        paddingVertical: height(1),
        borderBottomWidth: 1,
        borderColor: '#fff'
    },
    headerText: {
        fontSize: 28,
        fontWeight: 'bold',
        color: AppColors.white,
    },
    quizContent: {
        padding: 20,
        flex: 1,
        justifyContent: 'center',
    },
    questionText: {
        fontSize: 20,
        fontWeight: '600',
        marginBottom: height(3),
        color: '#fff',
    },
    optionButton: {
        padding: height(2),
        borderRadius: 10,
        marginBottom: height(1.5),
        alignItems: 'center',
    },
    optionText: {
        fontSize: 18,
        color: '#fff',
    },
    submitButton: {
        backgroundColor: AppColors.primary,
        paddingVertical: height(2),
        borderRadius: 20,
        alignItems: 'center',
        marginTop: height(3),
    },
    submitButtonText: {
        fontSize: 18,
        color: '#fff',
        fontWeight: '600',
    },
    resultView: {
        // marginTop: height(15),
        alignItems: 'center',
        justifyContent: 'center',
        flex: 1,
    },
    resultText: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#fff',
    },
    scoreText: {
        fontSize: 20,
        marginVertical: height(2),
        color: AppColors.secondary,
    },
    restartButton: {
        backgroundColor: AppColors.primary,
        paddingVertical: height(2),
        paddingHorizontal: width(10),
        borderRadius: 20,
        marginTop: height(3),
    },
    restartButtonText: {
        color: '#fff',
        fontSize: 18,
        fontWeight: '600',
    },
});
