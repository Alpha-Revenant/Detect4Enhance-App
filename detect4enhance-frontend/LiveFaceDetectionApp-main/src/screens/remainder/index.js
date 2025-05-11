import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Platform } from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';
import * as Notifications from 'expo-notifications';

const ReminderScreen = () => {
    const [date, setDate] = useState(new Date());
    const [showPicker, setShowPicker] = useState(false);
    const [reminderSet, setReminderSet] = useState(false);

    console.log("date", date);

    useEffect(() => {
        const requestPermissions = async () => {
            const { status } = await Notifications.requestPermissionsAsync();
            if (status !== 'granted') {
                alert('You need to enable notifications for this app to work.');
            }
        };

        requestPermissions();
    }, []);


    const onChange = (event, selectedDate) => {
        if (Platform.OS === 'android') {
            setShowPicker(false);
        }

        if (event.type === 'set') {
            if (selectedDate) {
                setDate(selectedDate);
            }
        } else {
            console.log('Date picker cancelled');
        }
    };


    const scheduleNotification = async () => {
        const hours = date.getHours();
        const minutes = date.getMinutes();
        const ampm = hours >= 12 ? 'PM' : 'AM';
        const formattedHours = hours % 12 || 12;
        const formattedMinutes = minutes < 10 ? `0${minutes}` : minutes;

        await Notifications.scheduleNotificationAsync({
            content: {
                title: 'Break Remainder',
                body: 'This is a Remainder for Break',
                sound: true,
            },
            trigger: {
                date: date,
            },
        });

        alert(`Break Remainder set for ${formattedHours}: ${formattedMinutes} ${ampm}`);
    };


    return (
        <View style={styles.container}>
            <Text style={styles.title}>Set Your Break Reminder</Text>
            <Text style={styles.subtitle}>
                Choose a time for your break, and we'll remind you when it's time!
            </Text>

            {/* Date/Time Picker */}
            <TouchableOpacity style={styles.pickerButton} onPress={() => setShowPicker(true)}>
                <Text style={styles.pickerButtonText}>Select Date & Time</Text>
            </TouchableOpacity>

            {/* Display selected date/time */}
            <Text style={styles.selectedTime}>
                Selected Time: {date.toLocaleString()}
            </Text>

            {/* Show the date/time picker */}
            {showPicker && (
                <DateTimePicker
                    value={date}
                    mode="time"
                    display="default"
                    onChange={onChange}
                    minimumDate={new Date()}
                />
            )}

            {/* Set Reminder Button */}
            <TouchableOpacity
                style={[styles.setReminderButton,
                    //  reminderSet && styles.disabledButton
                ]}
                onPress={scheduleNotification}
            // disabled={reminderSet}
            >
                <Text style={styles.setReminderButtonText}>
                    {reminderSet ? 'Reminder Set!' : 'Set Reminder'}
                </Text>
            </TouchableOpacity>
        </View>
    );
};

// Styles
const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
        backgroundColor: '#f5f5f5',
    },
    title: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#333',
        marginBottom: 10,
    },
    subtitle: {
        fontSize: 16,
        color: '#666',
        textAlign: 'center',
        marginBottom: 30,
    },
    pickerButton: {
        backgroundColor: '#6200ee',
        padding: 15,
        borderRadius: 10,
        marginBottom: 20,
    },
    pickerButtonText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: 'bold',
    },
    selectedTime: {
        fontSize: 18,
        color: '#333',
        marginBottom: 30,
    },
    setReminderButton: {
        backgroundColor: '#03dac6',
        padding: 15,
        borderRadius: 10,
        width: '100%',
        alignItems: 'center',
    },
    setReminderButtonText: {
        color: '#fff',
        fontSize: 18,
        fontWeight: 'bold',
    },
    disabledButton: {
        backgroundColor: '#ccc',
    },
});

export default ReminderScreen;