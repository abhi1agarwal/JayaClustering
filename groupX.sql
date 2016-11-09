-- phpMyAdmin SQL Dump
-- version 4.0.10deb1
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Nov 09, 2016 at 05:45 AM
-- Server version: 5.5.53-0ubuntu0.14.04.1
-- PHP Version: 5.5.9-1ubuntu4.20

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `groupX`
--

-- --------------------------------------------------------

--
-- Table structure for table `committee`
--

CREATE TABLE IF NOT EXISTS `committee` (
  `dept_id` varchar(10) NOT NULL,
  `committee_id` varchar(10) NOT NULL,
  `committee_name` varchar(50) NOT NULL,
  PRIMARY KEY (`dept_id`,`committee_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `committee`
--

INSERT INTO `committee` (`dept_id`, `committee_id`, `committee_name`) VALUES
('2 ', '1', 'DDPC'),
('4', '1', 'DDPC');

-- --------------------------------------------------------

--
-- Table structure for table `course`
--

CREATE TABLE IF NOT EXISTS `course` (
  `course_id` varchar(10) NOT NULL,
  `course_name` varchar(50) NOT NULL,
  `course_coordinator` varchar(50) NOT NULL,
  `course_instructor` varchar(50) NOT NULL,
  PRIMARY KEY (`course_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `course`
--

INSERT INTO `course` (`course_id`, `course_name`, `course_coordinator`, `course_instructor`) VALUES
('1', 'Theory Course 1', 'Mr. ABC', 'Mr. XYZ'),
('2', 'Theory Course 2', 'Mr.XYZ ', 'Mr.B K Sin'),
('3', 'Other Course 1', 'Mr. ABC', 'Mr.XYZ'),
('4', 'Other Course 2', 'Mr. ABC', 'Mr. XYZ'),
('5', 'Theory Course 3', 'Mr. YUI', 'Mr. ABC'),
('6', 'Other Courses 3', 'Mr. YUI', 'Mr. ABC');

-- --------------------------------------------------------

--
-- Table structure for table `courseregistration`
--

CREATE TABLE IF NOT EXISTS `courseregistration` (
  `reg_no` varchar(10) NOT NULL,
  `course_id` varchar(10) NOT NULL,
  `credits_enrolled` decimal(3,0) NOT NULL,
  `sem_no` decimal(2,0) NOT NULL,
  `sem_type` tinyint(1) NOT NULL,
  `academic_year` year(4) DEFAULT NULL,
  `progress` varchar(25) NOT NULL,
  `status` varchar(25) NOT NULL,
  PRIMARY KEY (`reg_no`,`course_id`),
  KEY `course_id` (`course_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `courseregistration`
--

INSERT INTO `courseregistration` (`reg_no`, `course_id`, `credits_enrolled`, `sem_no`, `sem_type`, `academic_year`, `progress`, `status`) VALUES
('20134065', '1', 4, 1, 1, 2016, '', ''),
('20134136', '1', 4, 1, 1, 2016, '', ''),
('20134136', '3', 12, 1, 1, 2016, '', ''),
('20134136', '4', 12, 2, 0, 2016, '', ''),
('20134148', '1', 4, 3, 1, 2016, 'ChairmanSDPC', 'approved'),
('20134148', '2', 6, 3, 1, 2016, 'ChairmanSDPC', 'approved'),
('20134148', '3', 8, 3, 1, 2016, 'ChairmanSDPC', 'approved'),
('20134148', '4', 8, 3, 1, 2016, 'ChairmanSDPC', 'approved');

-- --------------------------------------------------------

--
-- Table structure for table `courseresultmaster`
--

CREATE TABLE IF NOT EXISTS `courseresultmaster` (
  `reg_no` varchar(10) NOT NULL,
  `ay` decimal(4,0) NOT NULL,
  `sem_no` decimal(2,0) NOT NULL,
  `sem_type` tinyint(1) NOT NULL,
  `course_id` varchar(10) NOT NULL,
  `credits_earned` decimal(2,0) NOT NULL,
  `grade` varchar(5) NOT NULL,
  `result` varchar(10) NOT NULL,
  `entered_date` date NOT NULL,
  `enterede_by` varchar(50) NOT NULL,
  `verified_date` date NOT NULL,
  `verified_by` varchar(50) NOT NULL,
  PRIMARY KEY (`reg_no`,`course_id`),
  KEY `course_id` (`course_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `courseresultmaster`
--

INSERT INTO `courseresultmaster` (`reg_no`, `ay`, `sem_no`, `sem_type`, `course_id`, `credits_earned`, `grade`, `result`, `entered_date`, `enterede_by`, `verified_date`, `verified_by`) VALUES
('20134065', 2016, 1, 0, '1', 4, 'A', 'Pass', '2016-11-02', 'abs', '2016-11-24', 'ghi'),
('20134136', 2016, 1, 0, '1', 4, 'A', 'Pass', '2016-11-02', 'abc', '2016-11-23', 'xyz'),
('20134136', 2016, 1, 1, '3', 2, 'B+', 'Pass', '2016-11-02', 'abc', '2016-11-09', 'xyz'),
('20134136', 2017, 2, 0, '4', 3, 'F', 'Fail', '2016-11-03', 'rst', '2016-11-17', 'pqr');

-- --------------------------------------------------------

--
-- Table structure for table `currentsupervisor`
--

CREATE TABLE IF NOT EXISTS `currentsupervisor` (
  `reg_no` varchar(10) NOT NULL,
  `supervisor1_id` varchar(10) NOT NULL,
  `supervisor2_id` varchar(10) NOT NULL,
  PRIMARY KEY (`reg_no`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `currentsupervisor`
--

INSERT INTO `currentsupervisor` (`reg_no`, `supervisor1_id`, `supervisor2_id`) VALUES
('20134136', 'faculty1', '');

-- --------------------------------------------------------

--
-- Table structure for table `dakinout`
--

CREATE TABLE IF NOT EXISTS `dakinout` (
  `doc_id` varchar(10) NOT NULL,
  `date_in` datetime NOT NULL,
  `date_out` datetime NOT NULL,
  `reference` varchar(100) NOT NULL,
  `send_to` varchar(50) NOT NULL,
  `received_by` varchar(50) NOT NULL,
  `resend_outside` varchar(50) NOT NULL,
  PRIMARY KEY (`doc_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `department`
--

CREATE TABLE IF NOT EXISTS `department` (
  `dept_id` varchar(10) NOT NULL,
  `dept_name` varchar(50) NOT NULL,
  `hod` varchar(50) NOT NULL,
  PRIMARY KEY (`dept_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `department`
--

INSERT INTO `department` (`dept_id`, `dept_name`, `hod`) VALUES
('1', 'Civil Engineering', 'Dr. Duggal'),
('2', 'Chemical Engineering', ''),
('3', 'Mechanical Engineering', ''),
('4', 'Computer Science and Engineering', 'Dr. Neeraj Tyagi'),
('5', 'Electronics and Communication Engineering', 'Mr. Asim Mukherjee'),
('6', 'Electrical Engineering', 'Mr. HOD');

-- --------------------------------------------------------

--
-- Table structure for table `document`
--

CREATE TABLE IF NOT EXISTS `document` (
  `doc_id` varchar(10) NOT NULL,
  `member_id` varchar(10) NOT NULL,
  `sem_no` decimal(2,0) NOT NULL,
  `academic_year` decimal(4,0) NOT NULL,
  `application_type` varchar(20) NOT NULL,
  `date_of_upload` date NOT NULL,
  `date_of_final_approval` date NOT NULL,
  PRIMARY KEY (`doc_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `document`
--

INSERT INTO `document` (`doc_id`, `member_id`, `sem_no`, `academic_year`, `application_type`, `date_of_upload`, `date_of_final_approval`) VALUES
('1', '20134065', 1, 2016, '1', '2016-04-27', '0000-00-00'),
('2', '20134065', 1, 2016, '1', '2016-04-27', '0000-00-00'),
('3', '20134136', 0, 2016, '1', '2016-05-10', '0000-00-00'),
('4', '20134136', 0, 2016, '1', '2016-05-10', '0000-00-00'),
('5', '20134136', 0, 2016, '1', '2016-05-10', '0000-00-00'),
('6', '20134136', 0, 2016, '1', '2016-10-13', '0000-00-00'),
('7', '20134136', 0, 2016, '1', '2016-10-13', '0000-00-00'),
('8', '20134136', 1, 2016, '', '2016-10-13', '0000-00-00');

-- --------------------------------------------------------

--
-- Table structure for table `documentlookup`
--

CREATE TABLE IF NOT EXISTS `documentlookup` (
  `doc_type_id` int(11) NOT NULL AUTO_INCREMENT,
  `doc_type` varchar(50) NOT NULL,
  PRIMARY KEY (`doc_type_id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=4 ;

--
-- Dumping data for table `documentlookup`
--

INSERT INTO `documentlookup` (`doc_type_id`, `doc_type`) VALUES
(1, 'Bonafide'),
(2, 'Passport size Photo'),
(3, 'Transcript');

-- --------------------------------------------------------

--
-- Table structure for table `examinarpanel`
--

CREATE TABLE IF NOT EXISTS `examinarpanel` (
  `reg_no` varchar(10) NOT NULL,
  `type` varchar(50) NOT NULL,
  `faculty_id` varchar(10) NOT NULL,
  `role` varchar(50) NOT NULL,
  PRIMARY KEY (`reg_no`,`type`,`faculty_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `faculty`
--

CREATE TABLE IF NOT EXISTS `faculty` (
  `faculty_id` varchar(10) NOT NULL,
  `password` varchar(255) NOT NULL,
  `name` varchar(50) NOT NULL,
  `dept_id` varchar(10) NOT NULL,
  `designation` varchar(50) NOT NULL,
  `contact` decimal(15,0) NOT NULL,
  `mail_id` varchar(50) NOT NULL,
  `external` binary(1) NOT NULL,
  `affiliation` varchar(100) NOT NULL,
  `photo_path` varchar(255) NOT NULL,
  PRIMARY KEY (`faculty_id`),
  KEY `dept_id` (`dept_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `faculty`
--

INSERT INTO `faculty` (`faculty_id`, `password`, `name`, `dept_id`, `designation`, `contact`, `mail_id`, `external`, `affiliation`, `photo_path`) VALUES
('1998005', 'hello', 'V K Singh', '4', 'Professor', 9781246262, 'vks@mnnit.ac.in', '0', '', ''),
('2005109', 'hello', 'Amit Varshney', '4', 'Associate Professor', 987654321, 'av@mnnit.ac.in', '0', '', ''),
('faculty1', 'hello', 'S K Mittal', '4', 'Professor', 123565324, 'as@gmail.co', '1', 'none', './images/faculty1.jpg'),
('faculty2', 'hello', 'Anurag Varshney', '4', 'Associate Professor', 9782365368, 'av@mnnit.ac.in', 'N', '', '');

-- --------------------------------------------------------

--
-- Table structure for table `jobdocumentlookup`
--

CREATE TABLE IF NOT EXISTS `jobdocumentlookup` (
  `job_type_id` int(11) NOT NULL,
  `doc_type_id` int(11) NOT NULL,
  PRIMARY KEY (`doc_type_id`,`job_type_id`),
  KEY `job_type_id` (`job_type_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `jobdocumentlookup`
--

INSERT INTO `jobdocumentlookup` (`job_type_id`, `doc_type_id`) VALUES
(1, 1),
(1, 3);

-- --------------------------------------------------------

--
-- Table structure for table `joblookup`
--

CREATE TABLE IF NOT EXISTS `joblookup` (
  `job_type_id` int(11) NOT NULL AUTO_INCREMENT,
  `job_type` varchar(50) NOT NULL,
  PRIMARY KEY (`job_type_id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=3 ;

--
-- Dumping data for table `joblookup`
--

INSERT INTO `joblookup` (`job_type_id`, `job_type`) VALUES
(1, 'Getting attested Transcript'),
(2, 'Process to get Scholarship 1');

-- --------------------------------------------------------

--
-- Table structure for table `leave`
--

CREATE TABLE IF NOT EXISTS `leave` (
  `reg_no` varchar(10) NOT NULL,
  `leave_type` varchar(20) NOT NULL,
  `sem_no` decimal(2,0) NOT NULL,
  `sem_type` varchar(10) NOT NULL,
  `academic_year` decimal(4,0) NOT NULL,
  `from_date` date NOT NULL,
  `to_date` date NOT NULL,
  `no_of_days` decimal(2,0) NOT NULL,
  `status` varchar(25) NOT NULL,
  `address` varchar(255) NOT NULL,
  `applied_on` date NOT NULL,
  `progress` varchar(25) NOT NULL DEFAULT 'Supervisor',
  PRIMARY KEY (`reg_no`,`leave_type`,`from_date`,`to_date`),
  KEY `leave_type` (`leave_type`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `leave`
--

INSERT INTO `leave` (`reg_no`, `leave_type`, `sem_no`, `sem_type`, `academic_year`, `from_date`, `to_date`, `no_of_days`, `status`, `address`, `applied_on`, `progress`) VALUES
('20134065', '3', 1, 'Odd', 2016, '2016-11-17', '2016-11-19', 3, 'denied', 'kml', '2016-11-03', 'ConvenerDDPC'),
('20134136', '3', 1, 'Odd', 2016, '2016-11-08', '2016-11-09', 2, 'pending', '40, Shivpuri, Bulandshahr, UP-203001', '2016-11-06', 'Supervisor'),
('20134136', '3', 1, 'Odd', 2016, '2016-11-08', '2016-11-26', 19, 'pending', 'blah blah', '2016-11-08', 'Supervisor'),
('20134136', '3', 1, 'Odd', 2016, '2016-11-09', '2016-11-11', 3, 'pending', '40, Shivpuri, Bulandshahr, UP-203001', '2016-11-06', 'ConvenerDDPC'),
('20134136', '3', 1, 'Odd', 2016, '2016-11-09', '2016-11-14', 6, 'pending', 'hjghg', '2016-11-07', 'ConvenerDDPC'),
('20134136', '3', 1, 'Odd', 2016, '2016-11-18', '2016-11-28', 11, 'pending', 'ghj', '2016-11-07', 'ConvenerDDPC'),
('20134148', '3', 1, 'Odd', 2016, '2016-11-09', '2016-11-10', 2, 'pending', 'Gorakhpur', '2016-11-06', 'Supervisor'),
('20134148', '3', 0, '', 2016, '2016-11-17', '2016-11-19', 3, 'approved', 'KNGH', '2016-11-04', 'HOD'),
('20134171', '3', 1, 'Odd', 2016, '2016-11-16', '2016-11-18', 3, 'pending', 'New Delhi', '2016-11-06', 'Supervisor');

-- --------------------------------------------------------

--
-- Table structure for table `leavelookup`
--

CREATE TABLE IF NOT EXISTS `leavelookup` (
  `leave_type` varchar(20) NOT NULL,
  `leave_name` varchar(20) NOT NULL,
  `no_of_days` decimal(3,0) NOT NULL,
  PRIMARY KEY (`leave_type`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `leavelookup`
--

INSERT INTO `leavelookup` (`leave_type`, `leave_name`, `no_of_days`) VALUES
('1', 'Sick Leave', 10),
('2', 'Personal', 5),
('3', 'casual', 15);

-- --------------------------------------------------------

--
-- Table structure for table `meetattendance`
--

CREATE TABLE IF NOT EXISTS `meetattendance` (
  `meeting_no` varchar(10) NOT NULL,
  `member_id` varchar(10) NOT NULL,
  PRIMARY KEY (`meeting_no`,`member_id`),
  KEY `member_id` (`member_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `meeting`
--

CREATE TABLE IF NOT EXISTS `meeting` (
  `meeting_no` varchar(10) NOT NULL,
  `dept_id` varchar(10) NOT NULL,
  `committee_id` varchar(10) NOT NULL,
  `date` date NOT NULL,
  `time` time NOT NULL,
  `venue` varchar(50) NOT NULL,
  `type` varchar(50) NOT NULL,
  PRIMARY KEY (`meeting_no`,`dept_id`,`committee_id`),
  KEY `dept_id` (`dept_id`,`committee_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `meeting`
--

INSERT INTO `meeting` (`meeting_no`, `dept_id`, `committee_id`, `date`, `time`, `venue`, `type`) VALUES
('1', '4', '1', '2016-11-08', '10:10:10', 'CSED', 'type1');

-- --------------------------------------------------------

--
-- Table structure for table `meetingagendabrief`
--

CREATE TABLE IF NOT EXISTS `meetingagendabrief` (
  `meeting_no` varchar(10) NOT NULL,
  `agenda_id` varchar(10) NOT NULL,
  `agenda_name` varchar(100) NOT NULL,
  `description` text NOT NULL,
  PRIMARY KEY (`meeting_no`,`agenda_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `meetingagendabrief`
--

INSERT INTO `meetingagendabrief` (`meeting_no`, `agenda_id`, `agenda_name`, `description`) VALUES
('1', '1', 'agenda', 'blah                                                ');

-- --------------------------------------------------------

--
-- Table structure for table `meetingdocs`
--

CREATE TABLE IF NOT EXISTS `meetingdocs` (
  `meeting_no` varchar(10) NOT NULL,
  `meeting_minute` longblob NOT NULL,
  `meeting_notice_with_agenda` longblob NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `members`
--

CREATE TABLE IF NOT EXISTS `members` (
  `member_id` varchar(10) NOT NULL,
  `member_type` varchar(20) NOT NULL,
  `committee_id` varchar(10) NOT NULL,
  `dept_id` varchar(10) NOT NULL,
  `role` varchar(25) NOT NULL,
  PRIMARY KEY (`member_id`,`committee_id`,`dept_id`),
  KEY `dept_id` (`dept_id`,`committee_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `members`
--

INSERT INTO `members` (`member_id`, `member_type`, `committee_id`, `dept_id`, `role`) VALUES
('1998005', 'internal', '1', '4', 'HOD'),
('2005109', 'internal', '1', '4', 'ConvenerDDPC'),
('20134136', 'student', '1', '4', 'student'),
('20134148', 'Student', '1 ', '4 ', 'student'),
('faculty1', 'internal', '1', '4', 'Supervisor'),
('faculty2', 'internal', '1', '4', 'ChairmanSDPC');

-- --------------------------------------------------------

--
-- Table structure for table `notifications`
--

CREATE TABLE IF NOT EXISTS `notifications` (
  `id` int(11) NOT NULL,
  `issue_date` date DEFAULT NULL,
  `description` varchar(100) DEFAULT NULL,
  `target_group` text NOT NULL,
  `target_member` text NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `notifications`
--

INSERT INTO `notifications` (`id`, `issue_date`, `description`, `target_group`, `target_member`) VALUES
(1, '2016-04-26', 'hello this a new notification', '', ''),
(2, '2016-04-27', 'sample notif', '', ''),
(3, '2016-05-09', 'hello there is a meeting', '', ''),
(4, '2016-05-10', 'a meeting', '', ''),
(5, '2015-05-10', 'hi this is ayushi', '', ''),
(6, '2016-05-10', 'this is first meeting', '', ''),
(7, '2016-05-13', 'i am at home', '', ''),
(8, '0000-00-00', 'hello', '', ''),
(9, '2016-11-07', 'blah', 'admin', ''),
(10, '2016-11-07', 'student', 'student', ''),
(11, '2016-11-07', 'this', 'Supervisor', ''),
(12, '2016-11-08', 'test notif', 'admin', ''),
(13, '2016-11-08', 'test', 'admin', ''),
(14, '2016-11-08', 'another', 'admin', ''),
(15, '2016-11-08', 'another', 'faculty', ''),
(16, '2016-11-08', 'another', 'ChairmanSDPC', '');

-- --------------------------------------------------------

--
-- Table structure for table `othercourses`
--

CREATE TABLE IF NOT EXISTS `othercourses` (
  `course_id` varchar(10) NOT NULL,
  `min_credits` decimal(2,0) NOT NULL,
  `max_credits` decimal(2,0) NOT NULL,
  PRIMARY KEY (`course_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `othercourses`
--

INSERT INTO `othercourses` (`course_id`, `min_credits`, `max_credits`) VALUES
('3', 8, 20),
('4', 8, 20),
('6', 8, 12);

-- --------------------------------------------------------

--
-- Table structure for table `partfullstatus`
--

CREATE TABLE IF NOT EXISTS `partfullstatus` (
  `reg_no` varchar(10) NOT NULL,
  `reg_status` varchar(20) NOT NULL,
  `date_of_modification` datetime NOT NULL,
  `reason` varchar(255) NOT NULL,
  `supervisor_comment` varchar(255) NOT NULL,
  `progress` varchar(25) NOT NULL,
  `status` varchar(25) NOT NULL,
  PRIMARY KEY (`reg_no`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `partfullstatus`
--

INSERT INTO `partfullstatus` (`reg_no`, `reg_status`, `date_of_modification`, `reason`, `supervisor_comment`, `progress`, `status`) VALUES
('20134136', 'Full-Time', '2016-11-04 00:00:00', 'thiS_that', 'my comment is this', 'ChairmanSDPC', 'approved');

-- --------------------------------------------------------

--
-- Table structure for table `rolelookup`
--

CREATE TABLE IF NOT EXISTS `rolelookup` (
  `role_id` varchar(25) NOT NULL,
  `role_name` varchar(50) NOT NULL,
  PRIMARY KEY (`role_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `rolelookup`
--

INSERT INTO `rolelookup` (`role_id`, `role_name`) VALUES
('ChairmanSDPC', 'SDPC Chairman'),
('ChairmanSenate', 'Senate Chairman'),
('ConvenerDDPC', 'DDPC Convener'),
('CourseCoordinator', 'Course Coordinator'),
('ExternalMemberSRC', 'External Member of SRC'),
('HOD', 'Head of Department'),
('InternalMemberSRC', 'Internal Member of SRC'),
('student', 'Student'),
('Supervisor', 'Supervisor');

-- --------------------------------------------------------

--
-- Table structure for table `src`
--

CREATE TABLE IF NOT EXISTS `src` (
  `reg_no` varchar(10) NOT NULL,
  `src_int_id` varchar(10) NOT NULL,
  `src_ext_id` varchar(10) NOT NULL,
  `supervisor1_id` varchar(10) NOT NULL,
  `supervisor2_id` varchar(10) NOT NULL,
  PRIMARY KEY (`reg_no`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `stipend`
--

CREATE TABLE IF NOT EXISTS `stipend` (
  `reg_no` varchar(10) NOT NULL,
  `month` decimal(2,0) NOT NULL,
  `year` decimal(4,0) NOT NULL,
  `date_sent` date NOT NULL,
  `stipend_amount` decimal(7,2) NOT NULL,
  PRIMARY KEY (`reg_no`,`month`,`year`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `studentmaster`
--

CREATE TABLE IF NOT EXISTS `studentmaster` (
  `reg_no` varchar(10) NOT NULL,
  `password` varchar(50) NOT NULL,
  `photo_path` varchar(50) NOT NULL,
  `category` varchar(10) NOT NULL,
  `program` varchar(20) NOT NULL,
  `name` varchar(50) NOT NULL,
  `father_name` varchar(50) NOT NULL,
  `address` varchar(100) NOT NULL,
  `contact_no` decimal(15,0) NOT NULL,
  `mail_id` varchar(30) NOT NULL,
  `hostel` varchar(50) NOT NULL,
  `gender` varchar(20) NOT NULL,
  `highest_qualification` varchar(100) NOT NULL,
  `nationality` varchar(25) NOT NULL,
  `admission_category_code` varchar(10) NOT NULL,
  `stipendiary` tinyint(1) NOT NULL,
  `program_type` varchar(25) NOT NULL,
  `program_category` varchar(25) NOT NULL,
  PRIMARY KEY (`reg_no`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `studentmaster`
--

INSERT INTO `studentmaster` (`reg_no`, `password`, `photo_path`, `category`, `program`, `name`, `father_name`, `address`, `contact_no`, `mail_id`, `hostel`, `gender`, `highest_qualification`, `nationality`, `admission_category_code`, `stipendiary`, `program_type`, `program_category`) VALUES
('20134065', 'hello', './images/20134065.jpg', 'Genral', 'B.Tech', 'Manish K Sinha', 'Rajesh Sinha Pathak', 'Mars', 1234567890, 'joker.ace@gmail.com', 'Tandon', 'M', 'AISSCE', 'Indian', '', 0, '', ''),
('20134136', 'gurha', './images/20134136.jpg', 'General', 'BTech', 'Ayushi Gurha', 'S G Gurha', 'G-5, KNGH', 9410671505, 'ayushigurha@gmail.com', 'KNGH', 'Female', 'AISCCE', 'Indian', '444', 0, 'Btech', '5005'),
('20134148', 'hello', '', '', '', 'Aishwarya Tripathi', 'R P M Tripathi', 'G_55, MNNIT', 987467846, 'aishmeow@gmail.com', 'KNGH', 'Female', 'Senior Secondary', 'Japanese', '11', 0, '', ''),
('20134171', 'hello', './images/20134171.jpg', 'General', 'B.Tech', 'Asim Krishna Prasad', 'Ajay Krishna Prasad', 'MNNIT, Allahabad', 8175843965, 'asimkprasad@gmail.com', 'Tandon', 'M', 'AISSCE', 'Indian', '', 1, '', '');

-- --------------------------------------------------------

--
-- Table structure for table `studentmincredit`
--

CREATE TABLE IF NOT EXISTS `studentmincredit` (
  `department` varchar(50) NOT NULL,
  `qualifying_degree` varchar(50) NOT NULL,
  `min_credit_to_earn` decimal(3,0) NOT NULL,
  `min_credit_through_course_work` decimal(2,0) NOT NULL,
  `min_credit_research_seminar` decimal(2,0) NOT NULL,
  `min_credit_through_project` decimal(2,0) NOT NULL,
  `credit_through_compre_exam` decimal(2,0) NOT NULL,
  `credit_through_soa` decimal(2,0) NOT NULL,
  `credit_through_research` decimal(2,0) NOT NULL,
  `min_duration` varchar(30) NOT NULL,
  `min_residence_full_time` varchar(30) NOT NULL,
  `max_duration_full_time` varchar(30) NOT NULL,
  `max_duration_part_time` varchar(30) NOT NULL,
  PRIMARY KEY (`department`,`qualifying_degree`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `studentmincredit`
--

INSERT INTO `studentmincredit` (`department`, `qualifying_degree`, `min_credit_to_earn`, `min_credit_through_course_work`, `min_credit_research_seminar`, `min_credit_through_project`, `credit_through_compre_exam`, `credit_through_soa`, `credit_through_research`, `min_duration`, `min_residence_full_time`, `max_duration_full_time`, `max_duration_part_time`) VALUES
('Engineering', 'B.Tech', 120, 32, 32, 32, 8, 8, 72, '3 years', '4 semesters', '6 years', '7 years'),
('Engineering', 'M. Tech', 80, 16, 16, 16, 8, 8, 48, '2 years', '4 semester', '6 years', '7 years'),
('Engineering', 'M.E.', 80, 16, 16, 16, 8, 8, 48, '2 years', '4 semesters', '6 years', '7 years'),
('Engineering', 'MCA', 120, 32, 32, 32, 8, 8, 72, '3 years', '4 semesters', '6 years', '7 years');

-- --------------------------------------------------------

--
-- Table structure for table `studentprogramdetails`
--

CREATE TABLE IF NOT EXISTS `studentprogramdetails` (
  `reg_no` varchar(10) NOT NULL,
  `date_of_comp_of_course_work` date NOT NULL,
  `credit_earn_course_work` decimal(2,0) NOT NULL,
  `credit_earn_thesis` decimal(2,0) NOT NULL,
  `date_of_comp` date NOT NULL,
  `date_of_soa` date NOT NULL,
  `date_of_open` date NOT NULL,
  `date_of_final_viva` date NOT NULL,
  `date_thesis_submission` date NOT NULL,
  `date_of_termination` date NOT NULL,
  `completed` tinyint(1) NOT NULL,
  `program_left` tinyint(1) NOT NULL,
  PRIMARY KEY (`reg_no`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `studentprogramdetails`
--

INSERT INTO `studentprogramdetails` (`reg_no`, `date_of_comp_of_course_work`, `credit_earn_course_work`, `credit_earn_thesis`, `date_of_comp`, `date_of_soa`, `date_of_open`, `date_of_final_viva`, `date_thesis_submission`, `date_of_termination`, `completed`, `program_left`) VALUES
('20134065', '2016-11-09', 9, 12, '2016-11-11', '2016-11-25', '2016-11-18', '2017-05-19', '2017-11-30', '2018-04-18', 0, 0),
('20134136', '2016-11-09', 10, 12, '0000-00-00', '2017-03-14', '2016-11-08', '2017-01-18', '2017-03-31', '2017-06-27', 0, 0);

-- --------------------------------------------------------

--
-- Table structure for table `studentregistration`
--

CREATE TABLE IF NOT EXISTS `studentregistration` (
  `reg_no` varchar(10) NOT NULL,
  `sem_no` decimal(2,0) NOT NULL,
  `sem_type` varchar(10) NOT NULL,
  `registration_by` varchar(50) NOT NULL,
  `date_of_reg` date NOT NULL,
  `remarks` text NOT NULL,
  `total_credits_registered` decimal(3,0) NOT NULL,
  PRIMARY KEY (`reg_no`,`sem_no`,`sem_type`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `studentregistration`
--

INSERT INTO `studentregistration` (`reg_no`, `sem_no`, `sem_type`, `registration_by`, `date_of_reg`, `remarks`, `total_credits_registered`) VALUES
('20134065', 1, 'Odd', 'Admin', '2016-04-05', '', 12),
('20134136', 1, 'Odd', 'Admin', '2016-10-13', '', 12),
('20134148', 1, 'Odd', 'Admin', '2016-11-01', '', 12),
('20134171', 1, 'Odd', 'Admin', '2016-04-05', '', 16);

-- --------------------------------------------------------

--
-- Table structure for table `studentthesisdetails`
--

CREATE TABLE IF NOT EXISTS `studentthesisdetails` (
  `reg_no` varchar(10) NOT NULL,
  `AOR` varchar(150) NOT NULL,
  `proposed_topic` varchar(150) NOT NULL,
  `final_topic` varchar(150) NOT NULL,
  `soa_report` longblob NOT NULL,
  PRIMARY KEY (`reg_no`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `studentthesisdetails`
--

INSERT INTO `studentthesisdetails` (`reg_no`, `AOR`, `proposed_topic`, `final_topic`, `soa_report`) VALUES
INSERT INTO `studentthesisdetails` (`reg_no`, `AOR`, `proposed_topic`, `final_topic`, `soa_report`) VALUES

-- --------------------------------------------------------

--
-- Table structure for table `supervisorhistory`
--

CREATE TABLE IF NOT EXISTS `supervisorhistory` (
  `reg_no` varchar(10) NOT NULL,
  `supervisor_id` varchar(10) NOT NULL,
  `date_of_allotment` date NOT NULL,
  `date_of_relieving` date NOT NULL,
  PRIMARY KEY (`reg_no`,`supervisor_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `supervisorhistory`
--

INSERT INTO `supervisorhistory` (`reg_no`, `supervisor_id`, `date_of_allotment`, `date_of_relieving`) VALUES
('20134136', 'faculty1', '2016-09-13', '0000-00-00'),
('20134148', 'faculty1', '2016-11-03', '0000-00-00');

-- --------------------------------------------------------

--
-- Table structure for table `theorycourses`
--

CREATE TABLE IF NOT EXISTS `theorycourses` (
  `course_id` varchar(10) NOT NULL,
  `total_credits` decimal(2,0) NOT NULL,
  PRIMARY KEY (`course_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `theorycourses`
--

INSERT INTO `theorycourses` (`course_id`, `total_credits`) VALUES
('1', 4),
('2', 6),
('5', 2);

--
-- Constraints for dumped tables
--

--
-- Constraints for table `committee`
--
ALTER TABLE `committee`
  ADD CONSTRAINT `committee_ibfk_1` FOREIGN KEY (`dept_id`) REFERENCES `department` (`dept_id`);

--
-- Constraints for table `courseregistration`
--
ALTER TABLE `courseregistration`
  ADD CONSTRAINT `courseRegistration_ibfk_1` FOREIGN KEY (`reg_no`) REFERENCES `studentmaster` (`reg_no`),
  ADD CONSTRAINT `courseRegistration_ibfk_2` FOREIGN KEY (`course_id`) REFERENCES `course` (`course_id`);

--
-- Constraints for table `courseresultmaster`
--
ALTER TABLE `courseresultmaster`
  ADD CONSTRAINT `courseResultMaster_ibfk_1` FOREIGN KEY (`reg_no`) REFERENCES `courseregistration` (`reg_no`),
  ADD CONSTRAINT `courseResultMaster_ibfk_2` FOREIGN KEY (`course_id`) REFERENCES `courseregistration` (`course_id`);

--
-- Constraints for table `currentsupervisor`
--
ALTER TABLE `currentsupervisor`
  ADD CONSTRAINT `currentSupervisor_ibfk_1` FOREIGN KEY (`reg_no`) REFERENCES `studentmaster` (`reg_no`);

--
-- Constraints for table `dakinout`
--
ALTER TABLE `dakinout`
  ADD CONSTRAINT `dakinout_ibfk_1` FOREIGN KEY (`doc_id`) REFERENCES `document` (`doc_id`);

--
-- Constraints for table `faculty`
--
ALTER TABLE `faculty`
  ADD CONSTRAINT `faculty_ibfk_1` FOREIGN KEY (`dept_id`) REFERENCES `department` (`dept_id`);

--
-- Constraints for table `jobdocumentlookup`
--
ALTER TABLE `jobdocumentlookup`
  ADD CONSTRAINT `jobdocumentlookup_ibfk_1` FOREIGN KEY (`doc_type_id`) REFERENCES `documentlookup` (`doc_type_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `jobdocumentlookup_ibfk_2` FOREIGN KEY (`job_type_id`) REFERENCES `joblookup` (`job_type_id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `leave`
--
ALTER TABLE `leave`
  ADD CONSTRAINT `leave_ibfk_1` FOREIGN KEY (`leave_type`) REFERENCES `leavelookup` (`leave_type`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `meetattendance`
--
ALTER TABLE `meetattendance`
  ADD CONSTRAINT `meetattendance_ibfk_1` FOREIGN KEY (`meeting_no`) REFERENCES `meeting` (`meeting_no`),
  ADD CONSTRAINT `meetattendance_ibfk_2` FOREIGN KEY (`member_id`) REFERENCES `members` (`member_id`);

--
-- Constraints for table `meeting`
--
ALTER TABLE `meeting`
  ADD CONSTRAINT `meeting_ibfk_1` FOREIGN KEY (`dept_id`, `committee_id`) REFERENCES `committee` (`dept_id`, `committee_id`),
  ADD CONSTRAINT `meeting_ibfk_2` FOREIGN KEY (`dept_id`, `committee_id`) REFERENCES `committee` (`dept_id`, `committee_id`);

--
-- Constraints for table `meetingagendabrief`
--
ALTER TABLE `meetingagendabrief`
  ADD CONSTRAINT `meetingagendabrief_ibfk_1` FOREIGN KEY (`meeting_no`) REFERENCES `meeting` (`meeting_no`);

--
-- Constraints for table `members`
--
ALTER TABLE `members`
  ADD CONSTRAINT `members_ibfk_1` FOREIGN KEY (`dept_id`, `committee_id`) REFERENCES `committee` (`dept_id`, `committee_id`);

--
-- Constraints for table `othercourses`
--
ALTER TABLE `othercourses`
  ADD CONSTRAINT `otherCourses_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`course_id`);

--
-- Constraints for table `partfullstatus`
--
ALTER TABLE `partfullstatus`
  ADD CONSTRAINT `partfullstatus_ibfk_1` FOREIGN KEY (`reg_no`) REFERENCES `studentmaster` (`reg_no`);

--
-- Constraints for table `src`
--
ALTER TABLE `src`
  ADD CONSTRAINT `src_ibfk_1` FOREIGN KEY (`reg_no`) REFERENCES `studentmaster` (`reg_no`);

--
-- Constraints for table `stipend`
--
ALTER TABLE `stipend`
  ADD CONSTRAINT `stipend_ibfk_1` FOREIGN KEY (`reg_no`) REFERENCES `studentmaster` (`reg_no`);

--
-- Constraints for table `studentprogramdetails`
--
ALTER TABLE `studentprogramdetails`
  ADD CONSTRAINT `studentProgramDetails_ibfk_1` FOREIGN KEY (`reg_no`) REFERENCES `studentmaster` (`reg_no`);

--
-- Constraints for table `studentregistration`
--
ALTER TABLE `studentregistration`
  ADD CONSTRAINT `studentregistration_ibfk_1` FOREIGN KEY (`reg_no`) REFERENCES `studentmaster` (`reg_no`);

--
-- Constraints for table `studentthesisdetails`
--
ALTER TABLE `studentthesisdetails`
  ADD CONSTRAINT `studentthesisdetails_ibfk_1` FOREIGN KEY (`reg_no`) REFERENCES `studentmaster` (`reg_no`);

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;